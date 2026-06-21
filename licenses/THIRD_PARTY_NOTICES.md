# Third-Party Notices

SafeShot includes third-party Python packages, build tools, selected model files,
and demo images from external research datasets. Before public distribution,
legal review should confirm that each dependency, checkpoint, dataset image, and
bundled asset may be redistributed in the intended countries and use cases.

This notice is provided for attribution and transparency. It is not legal advice
and is not a substitute for reviewing the authoritative upstream licenses, model
cards, dataset cards, and usage terms.

## Bundled Model Files

The SafeShot build process embeds selected VAE and scheduler files from:

* Model repository: `timbrooks/instruct-pix2pix`
* Source: https://huggingface.co/timbrooks/instruct-pix2pix
* Pinned revision used by SafeShot: `31519b5cb02a7fd89b906d88731cd4d6a7bbf88d`

Bundled files:

* `vae/config.json`
* `vae/diffusion_pytorch_model.safetensors`
* `scheduler/scheduler_config.json`

The macOS and Windows packaging processes embed these selected files into the
final offline application.

The upstream InstructPix2Pix repository states that pretrained checkpoints are
based on Stable Diffusion components and may be subject to additional checkpoint
terms. The authoritative upstream license, model card, Stable Diffusion license
terms, and any model-use restrictions should be reviewed and included with every
public release.

## Python Runtime and Packages

SafeShot is packaged as a standalone local application. Depending on the build
target, the release may include the Python runtime, PyInstaller bootloader files,
Python package files, package metadata, and transitive dependencies.

SafeShot depends on open-source Python packages including, but not limited to:

* `torch`
* `torchvision`
* `diffusers`
* `accelerate`
* `transformers`
* `safetensors`
* `gradio`
* `gradio_client`
* `huggingface_hub`
* `numpy`
* `Pillow`
* `scikit-image`
* `scipy`
* `PyInstaller`

Additional transitive or metadata-related packages may include:

* `filelock`
* `packaging`
* `PyYAML`
* `regex`
* `requests`
* `safehttpx`
* `tokenizers`
* `tqdm`
* `groovy`

Package licenses should be collected from the locked build environment and
shipped with every public release. The release should include applicable license
texts, copyright notices, package versions, and source/homepage information for
all bundled dependencies.

## Build and Packaging Tools

SafeShot builds may use the following packaging tools:

* `PyInstaller` for creating standalone application bundles
* Inno Setup for creating the Windows installer
* macOS `hdiutil` for DMG creation
* Apple `codesign` tools if signing macOS builds

If these tools are used in a public release workflow, their license terms and
documentation should be reviewed. PyInstaller, Python/CPython, and Inno Setup
license texts should be retained with release documentation where applicable.

## Demo Images

SafeShot demo images are sourced from the MagicBrush dataset and are processed
with SafeShot protection to demonstrate the visual quality of protected images.

* Dataset: MagicBrush
* Dataset repository: https://huggingface.co/datasets/osunlp/MagicBrush
* Project page: https://osu-nlp-group.github.io/MagicBrush/
* Paper: "MagicBrush: A Manually Annotated Dataset for Instruction-Guided Image Editing"
* License listed on Hugging Face: CC BY 4.0

Changes made by SafeShot:

* Selected MagicBrush images are protected using the SafeShot protection pipeline.
* The protected images contain SafeShot-generated perturbations.
* The demo images are used to show how images look after protection, not to claim
  ownership of the original MagicBrush images.

Attribution notice:

MagicBrush images are used under the MagicBrush dataset license. SafeShot does
not claim endorsement by the MagicBrush authors, the OSU NLP Group, or any
upstream dataset maintainers.

Before public release, confirm that every demo image included in the repository,
installer, website, poster, or report is compatible with the MagicBrush dataset
license and includes required attribution.

## Evaluation-Only Models and Assets

Some models, datasets, prompts, and generated outputs may have been used only for
research evaluation and are not embedded in the distributed SafeShot application.

Evaluation-only assets may include:

* Stable Diffusion v1.5 Inpainting, used as an evaluation attack model
* InstructPix2Pix evaluation outputs
* Test prompts and generated comparison outputs
* Research datasets or benchmark images used during development

Evaluation-only assets should not be assumed to be redistributable. If any
evaluation images, generated outputs, or benchmark assets are included in the
public repository, release package, report, poster, or demo, their source,
license, and usage permission should be documented separately.

## Research Acknowledgements

These citations are scholarly acknowledgements for the research ideas, model
families, datasets, and evaluation methods that SafeShot builds on. They are not
a substitute for dependency, dataset, or model license review.

Final SafeShot is based on or inspired by the following research directions:

* Hadi Salman, Alaa Khaddaj, Guillaume Leclerc, Andrew Ilyas, and Aleksander
  Madry. "Raising the Cost of Malicious AI-Powered Image Editing" (PhotoGuard).
  2023. https://arxiv.org/abs/2302.06588

* Ruoxi Chen, Haibo Jin, Yixin Liu, Jinyin Chen, Haohan Wang, and Lichao Sun.
  "EditShield: Protecting Unauthorized Image Editing by Instruction-guided
  Diffusion Models." ECCV 2024. https://arxiv.org/abs/2311.12066

* Jinsu Kim, Yunhun Nam, Minseon Kim, Sangpil Kim, and Jongheon Jeong.
  "BlurGuard: A Simple Approach for Robustifying Image Protection Against
  AI-Powered Editing." 2025. https://arxiv.org/abs/2511.00143

* Tim Brooks, Aleksander Holynski, and Alexei A. Efros. "InstructPix2Pix:
  Learning to Follow Image Editing Instructions." 2022.
  https://arxiv.org/abs/2211.09800

* Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bjorn
  Ommer. "High-Resolution Image Synthesis with Latent Diffusion Models." 2021.
  https://arxiv.org/abs/2112.10752

* Kai Zhang, Lingbo Mo, Wenhu Chen, Huan Sun, and Yu Su. "MagicBrush: A Manually
  Annotated Dataset for Instruction-Guided Image Editing." 2023.
  https://arxiv.org/abs/2306.10012

## Related Work Explored During Development

At the beginning of the project, the SafeShot team reproduced or attempted to
reproduce additional related image-protection methods, including Fawkes and
FaceLock. These early experiments did not produce successful results for the
final SafeShot threat model, and the team decided to move forward with a final
pipeline focused on PhotoGuard, EditShield, and BlurGuard.

No Fawkes or FaceLock code, weights, checkpoints, or assets are included in the
final distributed SafeShot application unless explicitly documented elsewhere.

Related early-exploration reference:

* Shawn Shan, Emily Wenger, Jiayun Zhang, Huiying Li, Haitao Zheng, and Ben Y.
  Zhao. "Fawkes: Protecting Privacy against Unauthorized Deep Learning Models."
  USENIX Security Symposium 2020. https://arxiv.org/abs/2002.08327

## SafeShot Release Labels

* **IP2P / EditShield** refers to SafeShot's InstructPix2Pix-oriented VAE latent
  protection path, related to the EditShield line of work.

* **SD / BlurGuard** refers to SafeShot's Stable Diffusion-oriented protection
  path with adaptive blur warmup and VAE latent optimization, related to the
  BlurGuard line of work.

These labels are descriptive research and implementation labels. They do not
imply endorsement by the authors of the referenced papers, dataset creators, or
maintainers of the upstream models and libraries.

## Release Checklist

Before publishing a SafeShot release, confirm that:

* All bundled Python packages have license texts and version information.
* The Python/CPython license is included if the interpreter is bundled.
* PyInstaller license and bootloader exception information are included where
  applicable.
* Inno Setup license information is included for Windows installer releases.
* The InstructPix2Pix model card, license, and Stable Diffusion-related terms
  have been reviewed.
* The MagicBrush dataset license and attribution requirements have been reviewed.
* Any demo image, evaluation image, generated output, or benchmark asset included
  in the release has a documented source and usage note.
* No evaluation-only model checkpoint or dataset asset is redistributed unless
  its license permits redistribution.
