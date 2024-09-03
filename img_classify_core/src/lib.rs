pub mod resource;

use anyhow::Error as E;
use candle_core::{DType, Device, Tensor};
use candle_nn::ops::softmax;
use candle_transformers::models::clip;
use tokenizers::Tokenizer;
use tracing::info;

pub fn tokenize_sequences(
    sequences: &[String],
    tokenizer: &Tokenizer,
    device: &Device,
) -> anyhow::Result<Tensor> {
    let pad_id = *tokenizer
        .get_vocab(true)
        .get("<|endoftext|>")
        .ok_or(E::msg("No pad token"))?;

    let mut tokens = vec![];

    for seq in sequences.iter() {
        let encoding = tokenizer.encode(seq.as_str(), true).map_err(E::msg)?;
        tokens.push(encoding.get_ids().to_vec());
    }

    let max_len = tokens.iter().map(|v| v.len()).max().unwrap_or(0);

    // Pad the sequences to have the same length
    for token_vec in tokens.iter_mut() {
        let len_diff = max_len - token_vec.len();
        if len_diff > 0 {
            token_vec.extend(vec![pad_id; len_diff]);
        }
    }

    let input_ids = Tensor::new(tokens, device)?;
    Ok(input_ids)
}

pub fn load_image<T: AsRef<std::path::Path>>(path: T, image_size: usize) -> anyhow::Result<Tensor> {
    let img = image::ImageReader::open(path)?.decode()?;
    let (height, width) = (image_size, image_size);
    let img = img.resize_to_fill(
        width as u32,
        height as u32,
        image::imageops::FilterType::Triangle,
    );

    let img = img.to_rgb8();

    let img = img.into_raw();
    let img = Tensor::from_vec(img, (height, width, 3), &Device::Cpu)?
        .permute((2, 0, 1))?
        .to_dtype(DType::F32)?
        .affine(2. / 255., -1.)?;
    // .unsqueeze(0)?;
    Ok(img)
}

pub fn load_images<T: AsRef<std::path::Path>>(
    paths: &[T],
    image_size: usize,
) -> anyhow::Result<Tensor> {
    let mut images = vec![];

    for path in paths {
        let tensor = load_image(path, image_size)?;
        images.push(tensor);
    }

    let images = Tensor::stack(&images, 0)?;

    Ok(images)
}

pub type ImgIdx = usize;
pub type SequenceIdx = usize;
pub struct Probability {
    pub seq: SequenceIdx,
    pub prob: f32,
}
pub struct Answer {
    pub img: ImgIdx,
    pub probs: Vec<Probability>,
}

pub fn forward(vec_imgs: &[String], sequences: &[String]) -> anyhow::Result<Vec<Answer>> {
    let config = clip::ClipConfig::vit_base_patch32();
    let model = resource::get_model(&config)?;
    let tokenizer = resource::get_tokenizer()?;

    let images = load_images(vec_imgs, config.image_size)?.to_device(&Device::Cpu)?;
    let input_ids = tokenize_sequences(sequences, &tokenizer, &Device::Cpu)?;

    let start = std::time::Instant::now();
    let (_logits_per_text, logits_per_image) = model.forward(&images, &input_ids)?;
    info!("forward costs:{:?}", start.elapsed());

    let softmax_image = softmax(&logits_per_image, 1)?;
    let softmax_image_vec = softmax_image.flatten_all()?.to_vec1::<f32>()?;
    info!("softmax_image_vec: {:?}", softmax_image_vec);

    let probability_per_image = softmax_image_vec.len() / vec_imgs.len();

    let mut answers = vec![];
    for (i, _img) in vec_imgs.iter().enumerate() {
        let start = i * probability_per_image;
        let end = start + probability_per_image;
        let probs: Vec<Probability> = (softmax_image_vec[start..end])
            .iter()
            .enumerate()
            .map(|(i, p)| Probability { seq: i, prob: *p })
            .collect();
        answers.push(Answer { img: i, probs });
    }
    Ok(answers)
}
