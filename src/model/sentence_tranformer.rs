use std::path::{Path, PathBuf};

use tch::{Tensor, Kind, Device, nn, Cuda, no_grad};
use rust_bert::Config;

use crate::model::bert::{Features, Bert};
use crate::model::pooling::{Pooling, PoolingConfig};


pub struct SentenceTransformer {
    pub bert: Bert,
    pub pooling: Pooling,
}

fn argsort<T: Ord>(v: &[T]) -> Vec<usize> {
    let mut idx = (0..v.len()).collect::<Vec<_>>();
    idx.sort_unstable_by(|&i, &j| v[i].cmp(&v[j]));
    idx
}

impl SentenceTransformer {
    pub fn new(model_path: &Path,
               device: Device
    ) -> failure::Fallible<SentenceTransformer> {
        let bert_model_path = model_path.join("0_BERT");
        let pooling_config_path = model_path.join("1_Pooling/config.json");
        let weights_path = model_path.join("model.ot");

        let bert = Bert::new(&bert_model_path.as_path(), None, None, device);
        let pooling = Pooling::new(&(&bert.vs.root() / "pooling"),
                                   &PoolingConfig::from_file(Path::new(&pooling_config_path)));

        Ok(SentenceTransformer { bert, pooling })
    }

    pub fn encode(&self,
                  sentences: Vec<&str>,
                  batch_size: i32) -> Vec<Vec<f64>>
    {
        let mut all_embeddings: Vec<Vec<f64>> = vec![];
        let length_sorted_idx = argsort(sentences.iter()
            .map(|input| input.len())
            .collect::<Vec<_>>()
            .as_ref());

        let length_sorted_idx = argsort(sentences.iter()
            .map(|input| input.len())
            .collect::<Vec<_>>()
            .as_ref());

        let batch_size = 8;
        for batch_idx in (0..sentences.len()).step_by(batch_size) {
            let mut batch_tokens = Vec::new();
            let batch_start = batch_idx;
            let batch_end = (batch_start + batch_size).min(sentences.len());

            let mut longest_seq = 0;

            for idx in &length_sorted_idx[batch_start..batch_end] {
                let sentence = sentences[*idx];
                let tokens = self.bert.tokenize(&sentence);
                longest_seq = longest_seq.max(tokens.len());
                batch_tokens.push(tokens);
            }

            let mut features = Features::default();


            let mut input_ids_feature = Vec::new();
            let mut token_type_ids_feature = Vec::new();
            let mut input_mask_feature = Vec::new();
            for text in &batch_tokens {
                let (input_ids, token_type_ids, input_mask, sentence_length) = self.bert.get_sentence_features(&text, longest_seq);
                input_ids_feature.push(Tensor::of_slice(&input_ids));
                token_type_ids_feature.push(Tensor::of_slice(&token_type_ids));
                input_mask_feature.push(Tensor::of_slice(&input_mask));
            }

            features.input_ids = Some(Tensor::stack(input_ids_feature.as_slice(), 0)
                .to(self.bert.vs.device()));

            features.token_type_ids = Some(Tensor::stack(token_type_ids_feature.as_slice(), 0)
                .to(self.bert.vs.device()));
            features.input_mask = Some(Tensor::stack(input_mask_feature.as_slice(), 0)
                .to(self.bert.vs.device()));

            let features = no_grad(|| {
                self.bert.forward_t(features)
            });

            let features = self.pooling.forward_t(features);
            let embeddings = features.sentence_embedding;

            all_embeddings.extend(Vec::<Vec<f64>>::from(&embeddings.unwrap()));
        }

        let reverting_order = argsort(&length_sorted_idx);


        let all_embeddings =
            reverting_order
                .iter()
                .map(|idx| all_embeddings[*idx].clone())
                .collect();


        all_embeddings
    }
}