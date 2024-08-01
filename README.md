Data Preprocessing: 
```sh
PUBLIC_DATASETS/acouslic-ai
    |-----images/
        |-----stacked_fetal_ultrasound
            |---0199616b-bdeb-4119-97a3-a5a3571bd641.mha
            |--- *.mha
        |-----unstacked_fetal_ultrasound
            |---0199616b-bdeb-4119-97a3-a5a3571bd641
                |----1.mha
                |----2.mha
                |...
```

### Generate Data Stats

```sh
python data_stats/generate_data_stats.py --csv_dir data_stats/artifacts --csv_filename ACOUSLIC_AI_data_stats.csv 
```

### Generate train-test split
written to artifacts/train_test_split
```sh
python data_stats/train_test_split.py
```