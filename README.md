## TensorFlow-Learning-Notes
some tensorflow notes by Leaves

### CONTENTS
- [Dataset API](https://github.com/LeavesLei/TensorFlow-Learning-Notes/blob/master/TensorFlow%20Dataset%20API.md)

### How can we plot .csv file downloaded form TensorBoard
[]()
#### patameter
- `csv_file`: `csv_file` is a dictionary which keys contain the type of csv file like test or train and values contain the path of csv file. Here is a example: `csv_file={"test": "C:/Users/Leaves/Desktop/run_test-tag-loss.csv", "train": "C:/Users/Leaves/Desktop/run_train-tag-loss.csv"}`
- `smooth_weight`: `smooth_weight` can make the curve more beatiful. default 0.
- `title`
- `ylable`
- `steps_or_runtime`: equals to 'step' means use Step as horizontal axis, others means use Wall time as horizontal axis
- `is_save`: save image. file name is `title` + ".png"
