use std::path::Path;

use tch::{Tensor, nn, Device, no_grad};
use tch::nn::VarStore;
use rust_bert::Config;
use rust_bert::bert::{BertModel, BertEmbeddings, BertConfig};

use rust_tokenizers::{BertTokenizer, BertVocab};
use rust_tokenizers::preprocessing::tokenizer::base_tokenizer::{Tokenizer, MultiThreadedTokenizer};
use tch::index::IndexOp;


#[derive(Debug)]
#[derive(Default)]
pub struct Features  {
    pub input_ids: Option<Tensor>,
    pub token_type_ids: Option<Tensor>,
    pub token_embeddings: Option<Tensor>,
    pub cls_token_embeddings: Option<Tensor>,
    pub input_mask: Option<Tensor>,
    pub sentence_embedding: Option<Tensor>,
    pub token_weights_sum: Option<Tensor>
}

impl Features {
    pub fn new(input_ids: Tensor,
               token_type_ids: Tensor,
               token_embeddings: Tensor,
               cls_token_embeddings: Tensor,
               input_mask: Tensor,
               sentence_embedding: Tensor,
               token_weights_sum: Tensor) -> Features {
        Features {
            input_ids: Some(input_ids),
            token_type_ids: Some(token_type_ids),
            token_embeddings: Some(token_embeddings),
            cls_token_embeddings:Some(cls_token_embeddings),
            input_mask: Some(input_mask),
            sentence_embedding: Some(sentence_embedding),
            token_weights_sum: Some(token_weights_sum) }
    }

    pub fn default() -> Features {
        Features {
            input_ids: None,
            token_type_ids: None,
            token_embeddings: None,
            cls_token_embeddings: None,
            input_mask: None,
            sentence_embedding: None,
            token_weights_sum: None }
    }
}

pub struct Bert {
    pub bert: BertModel<BertEmbeddings>,
    tokenizer: BertTokenizer,
    max_seq_length: i64,
    cls_token_id: i64,
    sep_token_id: i64,
    pub vs: VarStore
}

impl Bert {
    pub fn new(model_path: &Path,
               max_seq_length: Option<i64>,
               do_lower_case: Option<bool>,
               device: Device) -> Bert {
        let max_seq_length = if let Some(value) = max_seq_length {value} else { 128 };
        let do_lower_case = if let Some(value) = do_lower_case {value} else { true };

        let max_seq_length = if max_seq_length > 510 {
            warn!("Bert only allows a max_seq_length of 510 (512 with special tokens). Value will be set to 510");
            510
        } else {
            max_seq_length
        };

        let mut vs = nn::VarStore::new(device);

        let bert_config_path = model_path.join("config.json");
        let bert_vocab_path = model_path.join("vocab.txt");
        let weights_path = model_path.join("model.ot");

        let bert_config = BertConfig::from_file(bert_config_path.as_path());
        let bert: BertModel<BertEmbeddings> = BertModel::new(&(&vs.root() / "bert"), &bert_config);

        let tokenizer = BertTokenizer::from_file(bert_vocab_path.to_str().unwrap(), do_lower_case);
        let cls_token_id = tokenizer.convert_tokens_to_ids(&[String::from(BertVocab::cls_value())].to_vec())[0];
        let sep_token_id = tokenizer.convert_tokens_to_ids(&[String::from(BertVocab::sep_value())].to_vec())[0];

        vs.load(Path::new(&weights_path)).expect("Failed to load weights!");

        Bert {bert, tokenizer, max_seq_length, cls_token_id, sep_token_id,
            vs
        }
    }

    pub fn forward_t(&self, features: Features) -> Features {
        let (output_tokens, _, _, _) = no_grad(|| {
            self.bert.forward_t(
            Some(features.input_ids.as_ref().unwrap().shallow_clone()),
            Some(features.input_mask.as_ref().unwrap().shallow_clone()),
            Some(features.token_type_ids.as_ref().unwrap().shallow_clone()),
            None,
            None,
            &None,
            &None, //Some(features.input_mask.as_ref().unwrap().shallow_clone()),
            false).unwrap() });

        let cls_token = output_tokens.i((.., 0, ..));  //CLS token is first token

        Features {
            token_embeddings: Some(output_tokens),
            cls_token_embeddings: Some(cls_token),
            .. features }
    }

    pub fn tokenize(&self, text: &str) -> Vec<i64> {
        self.tokenizer.convert_tokens_to_ids(&self.tokenizer.tokenize(text))
    }

    pub fn tokenize_multithreaded(&self, text_list: Vec<&str>) -> Vec<Vec<i64>> {
        MultiThreadedTokenizer::tokenize_list(&self.tokenizer, text_list)
            .iter()
            .map(|sentence_tokens| self.tokenizer.convert_tokens_to_ids(sentence_tokens))
            .collect()
    }

    pub fn get_sentence_features(&self, tokens: &[i64], pad_seq_length: usize) -> (Vec<i64>, Vec<i64>, Vec<i64>, Vec<i64>) {

        let mut pad_seq_length = pad_seq_length.min(self.max_seq_length as usize);

        let tokens = if pad_seq_length < tokens.len() {
            &tokens[.. pad_seq_length as usize]
        } else {
            &tokens
        };

        let mut input_ids: Vec<i64> = vec![self.cls_token_id]
            .into_iter()
            .chain(tokens.to_vec().into_iter())
            .chain(vec![self.sep_token_id])
            .collect();
        let sentence_length = input_ids.len() ;

        pad_seq_length += 2;

        let mut token_type_ids = vec![0; input_ids.len()];
        let mut input_mask = vec![1; input_ids.len()];

        // Zero-pad up to the sequence length. Bert: Pad to the right
        let padding = vec![0; pad_seq_length as usize - input_ids.len()];
        input_ids.extend(&padding);
        token_type_ids.extend(&padding);
        input_mask.extend( &padding);

        assert_eq!(input_ids.len(), pad_seq_length);
        assert_eq!(input_mask.len(), pad_seq_length);
        assert_eq!(token_type_ids.len(), pad_seq_length);

        (input_ids, token_type_ids, input_mask, vec![sentence_length as i64])
    }
}