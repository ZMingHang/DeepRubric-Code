# DeepRubric Project Page

Static project page for the DeepRubric paper, published from this directory by GitHub Pages. No build step or runtime dependencies are required.

## Preview

From this directory, start a local preview:

```bash
python -m http.server 8123
```

Then visit:

```text
http://127.0.0.1:8123/
```

## Deploy

The repository's GitHub Pages source is `main` / `docs`. Changes to this directory are published after the Pages build succeeds.

## Page Files

- `index.html`: paper information, figures, results, resource links, and BibTeX.
- `styles.css`: responsive layout and local font/icon definitions.
- `script.js`: mobile navigation, section tracking, citation copying, and figure viewer.
- `assets/`: paper figures, local fonts, and icons. Keep these paths relative for GitHub Pages hosting.

The page remains readable with JavaScript disabled. Figure links open the original images, and the citation remains selectable.

## Third-Party Assets

- DM Sans and Instrument Serif are distributed under the SIL Open Font License. Their license files are included in `assets/fonts/`.
- Icons are from Lucide 0.468.0. Its license is included in `assets/icons/LICENSE`.
