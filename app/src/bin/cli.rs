use clap::Parser;
use tracing::info;

#[derive(Parser)]
struct Args {
    #[arg(long, use_value_delimiter = true)]
    images: Vec<String>,

    #[arg(long, use_value_delimiter = true)]
    sequences: Option<Vec<String>>,

    #[arg(long)]
    config: Option<String>,
}

pub fn main() -> anyhow::Result<()> {
    std::env::set_var("RUST_BACKTRACE", "full");
    let args = Args::parse();
    tracing_subscriber::fmt::init();

    let seq = args.sequences.unwrap_or(vec![
        "a cycling race".to_string(),
        "a photo of two cats".to_string(),
        "a robot holding a candle".to_string(),
    ]);
    let answers = img_classify_core::forward(&args.images, &seq)?;
    for a in answers {
        let img = a.img;
        info!("\n\nResults for image: {}\n", &args.images[img]);
        for p in a.probs.iter() {
            info!("Probability: {:.4}% Text: {} ", p.prob * 100.0, seq[p.seq]);
        }
    }
    Ok(())
}
