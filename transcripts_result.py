import cv2 #type: ignore
from table_header_row import TableAreaJson # type: ignore
from table_title_method_2 import TableHeaderRecognized
from recognized_table_areas_text import TabelAreasTextRecoginized
import json

image_dir = r"C:\Users\Admin\Downloads\AI_DETECT_MODEL_DOCUMENT\pagesPdf\page2_DTH.jpg"
# r"C:\Users\Admin\Downloads\AI_DETECT_MODEL_DOCUMENT\pagesPdf\page2_NMQ.jpg"

image = cv2.imread(image_dir)
model = TabelAreasTextRecoginized('page2.jpg')
table_image = model.table_image(image)
text_pages = model.recognize_table_areas_text(image)
model_header = TableHeaderRecognized(list(text_pages[0])[0])
_, header_arrays = model_header.predict_title_value()
filtered_array, result_arrays = model_header.predict_title_value()
print("Title table:",result_arrays)
model_table_json = TableAreaJson(list(text_pages[0])[0])
row_data = model_table_json.main_json()
for row in row_data:
    print(row)
header = [{'th': element['text']} for element in header_arrays]
json_result = json.dumps({'header': header, 'rows' : row_data}, ensure_ascii=False, indent=2,separators=(',', ': '))
print("Header: ", header)
