
import csv
from scipy.stats import mode

jeff, vgg = 'jeffnet_batchnorm_results.csv', 'top_5_predictions_vgg_run1.test.csv'

jeff_preds_list = []
v_preds_list = []
new_preds = []

with open(jeff) as j_test, open(vgg) as v_test:
  jeff_preds, v_preds = csv.reader(j_test), csv.reader(v_test)

  header_seen = False
  for row in jeff_preds:
    if not header_seen:
      header_seen = True
      continue

    jeff_preds_list.append(row)

  header_seen = False
  for row in v_preds:
    if not header_seen:
      header_seen = True
      continue
    v_preds_list.append(row)


for i in xrange(len(jeff_preds_list)):
  jeff_row = jeff_preds_list[i]
  v_preds_row = v_preds_list[i]
  
  if (v_preds_row[1] == jeff_row[1]):
    new_preds.append(v_preds_row)
    continue

  test_file = [jeff_row[0]]
  jeff_row = [int(x) for x in jeff_row[1:]]
  v_preds_row = [int(x) for x in v_preds_row[1:]]
  
  modal = mode(jeff_row + v_preds_row)
  count = modal.count[0]
  modal_val = modal.mode[0]
  if (count > 1):
    test_file.append(modal_val)
    for v in v_preds_row:
      if v != modal_val:
        test_file.append(v)
  else:
    test_file.extend(v_preds_row)

  new_preds.append(test_file)


  



