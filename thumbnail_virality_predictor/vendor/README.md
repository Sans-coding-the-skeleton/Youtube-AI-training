# Third-Party Dependencies (Vendor Folder)

**Requirements**: As per project specifications, all external, non-author code must be cleanly isolated and separated from the author's code.

In modern Python development, third-party libraries and code are managed by the environment (using virtual environments like `venv`) and the package manager `pip`, rather than manually copying `.py` files into a `/vendor/` or `/lib/` folder. This project strictly complies with this practice.

The `requirements.txt` specifies all third-party code packages:
1. `torch` and `torchvision` (PyTorch for Neural Network implementation)
2. `flask` (For the Web API and HTTP server)
3. `yt_dlp` (For interacting with the Youtube Data API and fetching metadata)
4. `Pillow` (For image processing)

No external CSS/JS libraries (like Bootstrap, Tailwind, or React) are used. The sole CSS dependency is a Google Font API call for the *Inter* font, configured purely through a standard `<link>` header rather than local files.

All code contained in the `/src` folder was authored, modified, and built by the project author.
