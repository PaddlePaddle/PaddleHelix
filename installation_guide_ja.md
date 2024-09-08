# インストールガイド

## 前提条件

* OSサポート: Windows, Linux, OSX
* Pythonバージョン: 3.6, 3.7

## 依存関係

| 名前         | バージョン |
| ------------ | ---- |
| numpy        | - |
| pandas       | - |
| networkx     | - |
| paddlepaddle | \>=2.0.0rc0 |
| pgl          | \>=2.1 |
| rdkit        | - |
| sklearn      | - |

（「-」は特定のバージョン要件がないことを意味します）

## インストール手順
PaddleHelixは、バージョン2.0.0rc0以上の`paddlepaddle`に依存しており、`rdkit`は`pip`を使用して直接インストールできないため、新しい環境を作成することをお勧めします。詳細な手順は以下の通りです：

1. condaがインストールされていない場合は、まずこのウェブサイトを参照してインストールしてください：

  https://docs.conda.io/projects/conda/en/latest/user-guide/install/

2. condaを使用して新しい環境を作成します：

```bash
conda create -n paddlehelix python=3.7  
```

3. 作成した環境をアクティブにします：

```bash
conda activate paddlehelix
```

4. condaを使用して`rdkit`をインストールします：

```bash
conda install -c conda-forge rdkit
```
5. `paddlepaddle`の適切なバージョンをインストールします。インストールするバージョンは、PaddleHelixを実行するデバイス（CPU/GPU）に応じて選択します。

    GPUバージョンの`paddlepaddle`を使用する場合は、次のコマンドを実行します：

    ```bash
    python -m pip install paddlepaddle-gpu -f https://paddlepaddle.org.cn/whl/stable.html
    ```

    または、CPUバージョンの`paddlepaddle`を使用する場合は、次のコマンドを実行します：

    ```bash
    python -m pip install paddlepaddle -i https://mirror.baidu.com/pypi/simple
    ```

    `paddlepaddle`のバージョンは**2.0**以上である必要があります。
    `paddlepaddle`の[公式ドキュメント](https://www.paddlepaddle.org.cn/documentation/docs/en/2.0-rc1/install/index_en.html)を参照して、詳細なインストールガイドを確認してください。

6. `pip`を使用して`PGL`をインストールします：
   
```bash
pip install pgl
```

7. `pip`を使用してPaddleHelixをインストールします：

```bash
pip install paddlehelix
```

8. インストールが完了しました！

### 注意
作業が終わったら、conda環境を非アクティブにするには、次のコマンドを実行します：

```bash
conda deactivate
```
