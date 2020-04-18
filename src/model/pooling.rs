use std::f32;
use serde::{Deserialize, Serialize};
use tch::{Tensor, nn, Kind};
use rust_bert::Config;
use crate::model::bert::Features;

#[derive(Debug, Serialize, Deserialize)]
pub struct PoolingConfig {
    pub word_embedding_dimension: i16,
    pub pooling_mode_cls_token: Option<bool>,
    pub pooling_mode_max_tokens: Option<bool>,
    pub pooling_mode_mean_tokens: Option<bool>,
    pub pooling_mode_mean_sqrt_len_tokens: Option<bool>
}

impl Config<PoolingConfig> for PoolingConfig {}

#[derive(Debug)]
pub struct Pooling {
    pooling_mode_cls_token: bool,
    pooling_mode_max_tokens: bool,
    pooling_mode_mean_tokens: bool,
    pooling_mode_mean_sqrt_len_tokens: bool,
    pooling_output_dimension: i16
}



impl Pooling {
    pub fn new(_p: &nn::Path,
               config: &PoolingConfig) -> Pooling {

        let pooling_mode_cls_token = if let Some(value) = config.pooling_mode_cls_token { value } else { false };
        let pooling_mode_max_tokens = if let Some(value) = config.pooling_mode_max_tokens { value } else { false };
        let pooling_mode_mean_tokens = if let Some(value) = config.pooling_mode_mean_tokens { value } else { true };
        let pooling_mode_mean_sqrt_len_tokens = if let Some(value) = config.pooling_mode_mean_sqrt_len_tokens { value } else { false };


        let pooling_mode_multiplier: i16 = (pooling_mode_cls_token ||
            pooling_mode_max_tokens ||
            pooling_mode_mean_tokens ||
            pooling_mode_mean_sqrt_len_tokens) as i16;
        let pooling_output_dimension = pooling_mode_multiplier * config.word_embedding_dimension;

        Pooling { pooling_mode_cls_token,
            pooling_mode_max_tokens,
            pooling_mode_mean_tokens,
            pooling_mode_mean_sqrt_len_tokens,
            pooling_output_dimension }
    }


    pub fn forward_t(&self,
                   features: Features) -> Features {

        let mut output_vectors = Vec::new();

        //      Pooling strategy
        if self.pooling_mode_cls_token {
            output_vectors.push(features.cls_token_embeddings.as_ref().unwrap().shallow_clone());
        }


        if self.pooling_mode_max_tokens {
            let input_mask_expanded = features.input_mask
                .as_ref()
                .unwrap()
                .unsqueeze(-1)
                .expand(&features.token_embeddings.as_ref().unwrap().size(), false)
                .to_kind(Kind::Float);
            let token_embeddings = features.token_embeddings
                .as_ref()
                .unwrap()
                .where1(&input_mask_expanded.ne(0), &Tensor::from(f32::MIN));

            let max_over_time = token_embeddings.max2(1, false).0;
            &output_vectors.push(max_over_time);
            }


        if self.pooling_mode_mean_tokens || self.pooling_mode_mean_sqrt_len_tokens {

            let input_mask_expanded = features.input_mask
                    .as_ref()
                    .unwrap()
                    .unsqueeze(-1)
                    .expand(&features.token_embeddings.as_ref().unwrap().size(), false)
                    .to_kind(Kind::Float);
            let sum_embeddings = Tensor::sum1(&(features.token_embeddings.as_ref().unwrap() * input_mask_expanded.as_ref()),
                                              &[1],
                                              false,
                                              Kind::Float);

            //    #If tokens are weighted (by WordWeights layer), feature 'token_weights_sum' will be present
            let mut sum_mask = match &features.token_weights_sum {
                Some(value) => value.shallow_clone().unsqueeze(-1).expand(&sum_embeddings.size(), false),
                None => input_mask_expanded.sum1(&[1], false, Kind::Float)
            };


            let sum_mask = sum_mask.clamp_min_(1e-9);

            if self.pooling_mode_mean_tokens {
                output_vectors.push(&sum_embeddings / &sum_mask);
            }

            if self.pooling_mode_mean_sqrt_len_tokens {
                output_vectors.push(&sum_embeddings / &sum_mask.sqrt())
            }

            &output_vectors
        } else {
            &output_vectors
        };

        let output_vector = Tensor::cat(&output_vectors, 1);
        Features {
            sentence_embedding: Some(output_vector),
            .. features
        }
    }
}