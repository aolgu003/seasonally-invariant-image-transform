# UML Use Case Diagrams

Use case diagrams documenting the user-facing functionality of each top-level script.

---

## `createTiledDataset.py`

Crops large orthorectified images into fixed-size tiles for use as training/validation data.

```mermaid
graph TB
    User["User"]

    subgraph CTD ["createTiledDataset.py"]
        direction TB

        UC_main(["Create Tiled Dataset"])

        subgraph required ["Required Arguments"]
            UC_indir(["Specify Input Directory\n─────────────────\n--raw_data_dir"])
            UC_outdir(["Specify Output Directory\n─────────────────\n--save_data_dir"])
        end

        subgraph optional ["Optional Arguments (have defaults)"]
            UC_ext(["Set File Extension\n─────────────────\n--file_extension\ndefault: png"])
            UC_overlap(["Set Overlap Ratio\n─────────────────\n--overlap_ratio\ndefault: 0.2"])
            UC_width(["Set Crop Width\n─────────────────\n--crop_width\ndefault: 600 px"])
            UC_height(["Set Crop Height\n─────────────────\n--crop_height\ndefault: 600 px"])
        end

        subgraph behavior ["System Behavior"]
            UC_mkdir(["Create Output Directory\nif Not Exists"])
            UC_discover(["Discover Image Files\nin Input Directory"])
            UC_crop(["Crop Each Image\ninto Tiles"])
            UC_save(["Save Tiles\nwith Indexed Filenames\n{name}_{000000}.png"])
        end
    end

    User -->|initiates| UC_main

    UC_main -->|"&lt;&lt;include&gt;&gt;"| UC_indir
    UC_main -->|"&lt;&lt;include&gt;&gt;"| UC_outdir

    UC_main -. "&lt;&lt;extend&gt;&gt;" .-> UC_ext
    UC_main -. "&lt;&lt;extend&gt;&gt;" .-> UC_overlap
    UC_main -. "&lt;&lt;extend&gt;&gt;" .-> UC_width
    UC_main -. "&lt;&lt;extend&gt;&gt;" .-> UC_height

    UC_main -->|"&lt;&lt;include&gt;&gt;"| UC_mkdir
    UC_main -->|"&lt;&lt;include&gt;&gt;"| UC_discover
    UC_discover -->|"&lt;&lt;include&gt;&gt;"| UC_crop
    UC_crop -->|"&lt;&lt;include&gt;&gt;"| UC_save
    UC_width -->|informs| UC_crop
    UC_height -->|informs| UC_crop
    UC_overlap -->|informs| UC_crop
    UC_ext -->|filters| UC_discover
```

### Key Behaviors

| Argument | Required | Default | Description |
|---|---|---|---|
| `--raw_data_dir` | Yes | — | Directory containing source images to tile |
| `--save_data_dir` | Yes | — | Directory where tiles will be written |
| `--file_extension` | No | `png` | File extension used to glob input images |
| `--overlap_ratio` | No | `0.2` | Fractional overlap between adjacent tiles (0 = no overlap) |
| `--crop_width` | No | `600` | Tile width in pixels |
| `--crop_height` | No | `600` | Tile height in pixels |

**Output filename pattern:** `{original_name}_{index:06d}.png`
**Side effect:** Output directory is created automatically if it does not exist.

---
