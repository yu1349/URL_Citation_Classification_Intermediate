# 渡邉が追加したもの

## データセット関連
- [me_data_full_split.ipynb](./me_data_full_split.ipynb)  
データ分割を行ったコード。
- [me_data_jsai_split.ipynb](./me_data_jsai_split.ipynb)  
JSAIまでのアノテーション量でデータ分割を行ったコード。Full_dataに対してのデータ分割を採用し、そのインデックスのなかから既にアノテーションされているものをフィルタリング。
- [me_data_full_preprocess.ipynb](./me_data_full_preprocess.ipynb)  
モデル入力のための前処理のコード。これは途中。

## 学習関連
- [me_full_tsunokake_baseline.py](./me_full_tsunokake_baseline.py)  
Tsunokakeらの手法の学習コード。