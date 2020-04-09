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
        "The population density was 1,573.9 people per square mile (606.7/km²).",
        "There were 1,446 housing units at an average density of 706.6 per square mile (272.3/km²).",
        "From 1991 to 2012, Bushnell was home to one of the largest Christian Music and Arts festivals in the world, known as the Cornerstone Festival.",
        "Each year around the 4th of July, 25,000 people from all over the world would descend on the small farm town to watch over 300 bands, authors and artists perform at the Cornerstone Farm Campgrounds.",
        "The festival was generally well received by locals, and businesses in the area would typically put up signs welcoming festival-goers to their town.",
        "As a result of the location of the music festival, numerous live albums and videos have been recorded or filmed in Bushnell, including the annual Cornerstone Festival DVD. ",
        "Cornerstone held its final festival in 2012 and no longer operates.",
        "Beginning in 1908, the Truman Pioneer Stud Farm in Bushnell was home to one of the largest horse shows in the Midwest.",
        "The show was well known for imported European horses.",
        "The Bushnell Horse Show features some of the best Belgian and Percheron hitches in the country. Teams have come from many different states and Canada to compete."
    ];

    let embedder = SentenceTransformer::new(
        Path::new("/path/to/sentence-tranformers-model/bert-base-nli-stsb-mean-tokens"),
        device)?;

    let embedings = &embedder.encode(sentences.to_vec(), 8);
    println!("{:?}", embedings);
    Ok(())
}