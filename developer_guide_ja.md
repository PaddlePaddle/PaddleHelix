# 開発者ガイド

PaddleHelixのアルゴリズムを変更する必要がある場合は、開発者モードでPaddleHelixを使用する必要があります。PaddleHelixのコアアルゴリズムはほとんどがPythonで実装されていますが、一部はC++でも実装されているため、`pip install --editable {pahelix_path}`を使用して簡単に開発することはできません。PaddleHelixをローカルコンピュータで開発するには、以下のチュートリアルをお読みください。

1. まず、[インストールガイド](./installation_guide_ja.md)に従ってPaddleHelixの依存関係（paddlepaddle >= 2.0.0rc0、pgl >= 1.2.0）をインストールしてください。

2. 以前に`pip install paddlehelix`を使用してPaddleHelixの配布パッケージをインストールした場合は、アンインストールしてください：

    ```bash
    pip uninstall paddlehelix
    ```

3. PaddleHelixのソースコードリポジトリをローカルにクローンします。パスを「/path_to_your_repo/」と仮定します：

    ```bash
    git clone https://github.com/PaddlePaddle/PaddleHelix.git /path_to_your_repo/
    cd /path_to_your_repo/
    ```

4. 変更するアルゴリズムに応じて、4.1または4.2の手順に従ってください：

    4.1. LinearRNA
            
    LinearRNAのソースコードは「./c/pahelix/toolkit/linear_rna/linear_rna」にあります。必要に応じてC++のソースコードを調整できます。コードを変更した後、プロジェクトのルートディレクトリに戻り、以下のスクリプトを呼び出して再コンパイルしてください（環境にcmake >= 3.6およびg++ >= 4.8がインストールされていることを確認してください）：

    ```bash
    sh scripts/prepare.sh
    sh scripts/build.sh
    ```

    コンパイルが成功した後、以下のコマンドを使用してLinearRNAを正常にインポートできます：

    ```bash
    cd build
    python
    >>> import c.pahelix.toolkit.linear_rna.linear_rna as linear_rna
    ```

    4.2. その他のアルゴリズム

    PaddleHelixのLinearRNA以外のアルゴリズムはすべてPythonで実装されています。これらのアルゴリズムを変更したい場合は、「./pahelix」パスの下にある対応するファイルを見つけてから、「/path_to_your_repo/」をPython環境パスに追加してください：

    ```python
    import sys
    sys.path.append('/path_to_your_repo/')
    import pahelix
    ```

質問や提案がある場合は、[issue](https://github.com/PaddlePaddle/PaddleHelix/issues)を提出してください。できるだけ早く返信いたします。
