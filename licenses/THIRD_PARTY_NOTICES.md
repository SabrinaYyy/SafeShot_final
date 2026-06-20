# Third-Party Notices

SafeShot includes third-party Python packages and model files. Before public
distribution, legal review should confirm that each dependency and checkpoint
may be redistributed in the intended countries and use cases.

## Model Files

The build process uses VAE and scheduler files from:

- Model: `timbrooks/instruct-pix2pix`
- Revision: `31519b5cb02a7fd89b906d88731cd4d6a7bbf88d`
- Source: <https://huggingface.co/timbrooks/instruct-pix2pix>

The macOS and Windows packaging processes embed these files into the final
offline application.

The upstream repository states that pretrained checkpoints are based on Stable
Diffusion components and may be subject to additional checkpoint terms. Include
the authoritative upstream license and model-use terms here after legal review.

## Python Packages

SafeShot depends on open-source Python packages including PyTorch, Diffusers,
Hugging Face Hub, Gradio, scikit-image, SciPy, Pillow, NumPy, safetensors, and
PyInstaller. Package licenses should be collected from the locked build
environment and shipped with every public release.

## Research Acknowledgements

These citations are scholarly acknowledgements for the research ideas and model
families that SafeShot builds on. They are not a substitute for dependency or
model license review.

- Ruoxi Chen, Haibo Jin, Yixin Liu, Jinyin Chen, Haohan Wang, and Lichao Sun.
  "EditShield: Protecting Unauthorized Image Editing by Instruction-guided
  Diffusion Models." ECCV 2024.
  <https://arxiv.org/abs/2311.12066>
- Jinsu Kim, Yunhun Nam, Minseon Kim, Sangpil Kim, and Jongheon Jeong.
  "BlurGuard: A Simple Approach for Robustifying Image Protection Against
  AI-Powered Editing." 2025.
  <https://arxiv.org/abs/2511.00143>
- Tim Brooks, Aleksander Holynski, and Alexei A. Efros. "InstructPix2Pix:
  Learning to Follow Image Editing Instructions." 2022.
  <https://arxiv.org/abs/2211.09800>
- Hadi Salman, Alaa Khaddaj, Guillaume Leclerc, Andrew Ilyas, and Aleksander
  Madry. "Raising the Cost of Malicious AI-Powered Image Editing"
  (PhotoGuard). 2023.
  <https://arxiv.org/abs/2302.06588>
- Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bjorn
  Ommer. "High-Resolution Image Synthesis with Latent Diffusion Models." 2021.
  <https://arxiv.org/abs/2112.10752>

SafeShot release labels:

- **IP2P / EditShield** refers to SafeShot's InstructPix2Pix-oriented VAE latent
  protection path, related to the EditShield line of work.
- **SD / BlurGuard** refers to SafeShot's Stable Diffusion-oriented protection
  path with adaptive blur warmup and VAE latent optimization, related to the
  BlurGuard line of work.
