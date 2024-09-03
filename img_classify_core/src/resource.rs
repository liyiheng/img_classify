use anyhow::Result;

use candle_core::{DType, Device, Tensor};
use candle_nn::{ops::softmax, VarBuilder};
use candle_transformers::models::clip::{self, ClipConfig, ClipModel};

use rust_embed::Embed;
use tokenizers::Tokenizer;

#[cfg(feature = "fat")]
#[derive(Embed)]
#[folder = "resource/openai_clip_vit_base_patch32/"]
struct Asset;

#[cfg(not(feature = "fat"))]
struct Asset;
#[cfg(not(feature = "fat"))]
struct FakeEmbedFile<'a> {
    data: &'a [u8],
}
impl Asset {
    fn get(_: &str) -> Option<FakeEmbedFile> {
        None
    }
}

pub fn get_model(config: &ClipConfig) -> Result<ClipModel> {
    let data = if cfg!(feature = "fat") {
        let f = Asset::get("model.safetensors").ok_or(anyhow::format_err!("not found"))?;
        f.data.to_vec()
    } else {
        let builder = hf_hub::api::sync::ApiBuilder::new();
        let api = builder
            .with_endpoint("https://hf-mirror.com".to_string())
            .build()?;
        let api = api.repo(hf_hub::Repo::with_revision(
            "openai/clip-vit-base-patch32".to_string(),
            hf_hub::RepoType::Model,
            "refs/pr/15".to_string(),
        ));
        let v = api.get("model.safetensors")?;
        std::fs::read(v)?
    };

    let vb = VarBuilder::from_slice_safetensors(&data, DType::F32, &Device::Cpu)?;

    let model = clip::ClipModel::new(vb, config)?;
    Ok(model)
}
pub fn get_tokenizer() -> Result<Tokenizer> {
    let data = if cfg!(feature = "fat") {
        let f = Asset::get("tokenizer.json").ok_or(anyhow::format_err!("not found"))?;
        f.data.to_vec()
    } else {
        let builder = hf_hub::api::sync::ApiBuilder::new();
        let api = builder
            .with_endpoint("https://hf-mirror.com".to_string())
            .build()?;
        let api = api.repo(hf_hub::Repo::with_revision(
            "openai/clip-vit-base-patch32".to_string(),
            hf_hub::RepoType::Model,
            "refs/pr/15".to_string(),
        ));
        let v = api.get("tokenizer.json")?;
        std::fs::read(v).map_err(|e| anyhow::anyhow!("read file err:{}", e))?
    };
    Tokenizer::from_bytes(data).map_err(|e| anyhow::anyhow!("failed to create tokenizer, {}", e))
}
