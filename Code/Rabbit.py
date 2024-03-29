# Tiền xử lí: thực hiện Panthera và ghi lại
import pandas as pd
import csv
from Panthera import Panthera
class Rabbit:
  def __init__(self) -> None:
    self.call_function = False
    pass
  def get_file(self,function = None):
    # thực hiện các thao tác tạo file, lấy file,...
    if self.call_function:
      return "already"

    self.call_function = True

    if function == None:

      file_path = 'Data\\Data-for-Data-Mining-Project - Source.csv'

      file_path_save = 'Data\\Comment_Sentiment.csv'
      # pd.DataFrame(columns = ['ID','Tokenize Sentence', 'Label']).to_csv(file_path_save, header=True, index=False, mode = 'w')

      children_leopard = Panthera()

      df = pd.read_csv(file_path)

      header = True

      for i in range(df.shape[0]):
        sub_df = df.iloc[[i]]
        text = sub_df['Review Title'] + '. ' + sub_df['Review Content']
        text = text.values[0]
        data = {'ID':i,'Tokenize Sentence':children_leopard.execute(text), 'Label':sub_df['Label'].values[0]}
        print(data)

        # Lưu từ điển vào tệp CSV
        file_path = file_path_save
        my_dict = data
        with open(file_path, 'a', newline='') as csvfile:
            fieldnames = my_dict.keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            if header:
              writer.writeheader()
              header = False
            writer.writerow(my_dict)

    else:
       return function


if __name__ == "__main__":
  children_rabbit = Rabbit()
  children_rabbit.get_file()

# đã ghi