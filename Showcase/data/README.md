# Dataset placement

Raw datasets are not bundled in this public showcase.

Place the segmentation datasets like this before training or evaluation:

```text
data/
├─ TrainDataset/
│  └─ TrainDataset/
│     ├─ image/
│     └─ masks/
└─ TestDataset/
   └─ TestDataset/
      ├─ CVC-300/
      │  ├─ images/
      │  └─ masks/
      ├─ CVC-ClinicDB/
      ├─ CVC-ColonDB/
      ├─ ETIS-LaribPolypDB/
      └─ Kvasir/
```

The scripts expect the same naming convention used in the original project directory.
