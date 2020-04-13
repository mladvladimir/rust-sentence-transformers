# rust-sentence-transformers
Rust port of https://github.com/UKPLab/sentence-transformers
- Currently only supports loading BERT models, fine-tuned with original library.
- Both, model and pooling should be provided in directory structure compatible with original library.  
- Inference supported, only.
- Built on top of [rust-bert](https://github.com/guillaume-be/rust-bert).

```rust

use std::path::{Path, PathBuf};

use tch::{Tensor, Kind, Device, nn, Cuda, no_grad};

use rust_sentence_transformers::model::SentenceTransformer;

fn main() -> failure::Fallible<()> {

    let device = Device::Cpu;
    let sentences = [
        "Bushnell is located at 40°33′6″N 90°30′29″W (40.551667, -90.507921).",
        "According to the 2010 census, Bushnell has a total area of 2.138 square miles (5.54 km2), of which 2.13 square miles (5.52 km2) (or 99.63%) is land and 0.008 square miles (0.02 km2) (or 0.37%) is water.",
        "The town was founded in 1854 when the Northern Cross Railroad built a line through the area.",
        "Nehemiah Bushnell was the President of the Railroad, and townspeople honored him by naming their community after him.",
        "Bushnell was also served by the Toledo, Peoria and Western Railway, now the Keokuk Junction Railway.",
        "As of the census[6] of 2000, there were 3,221 people, 1,323 households, and 889 families residing in the city.",
    ];

    let embedder = SentenceTransformer::new(
        Path::new("/path/to/sentence-tranformers-model/bert-base-nli-stsb-mean-tokens"),
        device)?;

    let embedings = &embedder.encode(sentences.to_vec(), 8);
    println!("{:?}", embedings);
    Ok(())
}
```

### For more information, check:
- [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084)
- https://github.com/UKPLab/sentence-transformers
- https://github.com/guillaume-be/rust-bert