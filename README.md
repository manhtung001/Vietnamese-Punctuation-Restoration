# Vietnamese-Punctuation-Restoration
NLP Project VinBigData: Vietnamese Punctuation Restoration

## Demo
```
input: Là nhà phát triển bất động sản quốc tế với nhiều dự án từ biệt thự nhà phố căn hộ trung cao cấp Gamuda Land, Việt Nam tự hào kiến tạo những công trình có phong cách kiến trúc ấn tượng xứng tầm châu Á, không chỉ khách hàng đón nhận nhiệt tình mà còn được các chuyên gia bất động sản quốc tế đánh giá cao
```
```
output: Là nhà phát triển bất động sản quốc tế với nhiều dự án từ biệt thự, nhà phố, căn hộ trung cao cấp, Gamuda Land, Việt Nam tự hào kiến tạo những công trình có phong cách kiến trúc ấn tượng xứng tầm châu Á, không chỉ khách hàng đón nhận nhiệt tình mà còn được các chuyên gia bất động sản quốc tế đánh giá cao
```
## How to test this app

### Requirements

```
streamlit==1.6
numpy==1.19.5
altair==4.1.0
pandas==1.2.5
scipy==1.6.2
click==8
docx2txt
protobuf==3.20.*
pytorch-crf
transformers
underthesea==1.3.5a3
```

### Instructions

**0. Clone the folder app**

```sh
git clone https://github.com/manhtung001/Vietnamese-Punctuation-Restoration
```

**1. Contact me for phobert-base-gru_weights.pt**

**2. Install python package**
```sh
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```
```sh
pip install -r requirements.txt
```

**3. Run Streamlit**

```sh
streamlit run Vietnamese-Punctuation-Restoration/streamlit_app_flow.py
```