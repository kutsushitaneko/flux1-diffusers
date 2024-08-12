## FLUX.1 Diffusers サンプルコード

### このプロジェクトについて
Stable Diffusion の開発メンバーなどが立ち上げた __[Black Forest Labs](https://blackforestlabs.ai/)__ の画像生成AIモデル __FLUX.1__ が話題になっています。プロンプトへの追随性能もなかなか良く、人物も指や四肢の破綻の少ない高品質な画像を生成することができます。また、`[pro]`、`[dev]`、`[schnell]`の3つのモデルのうち __[dev]__ と __[schnell]__ はモデルウェイトが公開されていて OCI のようなクラウドやローカル環境でセルフホスティングすることができます。
<table>
<tr>
<td>

![ComfyUI_00077_.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/74534/f910cd82-fc67-93e0-90e9-86c03034a892.png)
</td>
<td>

![ComfyUI_00132_.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/74534/eef1e00f-8724-aab1-62c4-0a25beb9cbcd.png)
</td>
</tr>
</table>

このプロジェクトは、ローカルの Windows 11 PC、それも、__Nvidia GeForce RTX2060 VRAM:6GB__ という FLUX.1 のような巨大モデルには厳し過ぎる GPU 環境で様々な方法で動かしてみた経験をまとめた Qiita の記事（[画像生成AI FLUX.1 をBlack Forest Labs リファレンス実装、Diffusers、ComfyUI で動かしてみた（セルフホスト）](https://qiita.com/yuji-arakawa/items/fd4fd0c026ecfa664d97)）で紹介している 3つの方式
- __Black Forest Labs リファレンス実装（Model Offloading、NSFW検知あり）__
- __Hugging Face Diffusers（量子化かつ Model Offloading）__
- __ComfyUI__ 

のうちの Hugging Face Diffusers（量子化かつ Model Offloading ）のサンプルコードです。


## Hugging Face Diffusers
### FLUX.1 [dev]
#### デモ（Hugging Face Spaces）
https://huggingface.co/spaces/black-forest-labs/FLUX.1-dev
#### モデルカード
https://huggingface.co/black-forest-labs/FLUX.1-dev
### FLUX.1 [schnell]
#### デモ（Hugging Face Spaces）
https://huggingface.co/spaces/black-forest-labs/FLUX.1-schnell
#### モデルカード
https://huggingface.co/black-forest-labs/FLUX.1-schnell

### Diffusers を使ったサンプルコード
このサンプルコード（私、kutsushitaneko版）は、起動時の指定で Optimum Quanto を使った量子化に対応しています。Optimum Quanto は、Hugging Face Optimum の量子化バックエンドです。

https://github.com/huggingface/optimum-quanto

#### 前提条件
- CUDA 対応の GPU を搭載したハードウェア
- PyTorch が対応している CUDA Toolkit のバージョンがインストールされていること（[PyTorch と CUDA Toolkit のバージョンコンパチビリティの確認](https://pytorch.org/get-started/locally/)）

#### サンプルコードの取得
Github のリポジトリからクローンします
```
git clone https://github.com/kutsushitaneko/flux1-diffusers.git
cd flux1-diffusers
```

Python 仮想環境を作成します（推奨）。ここでは、venv を使います
```bash:Python仮想環境の作成
python3.10 -m venv .venv
source .venv/bin/activate
```
#### PyTorch のインストール
```bash:CUDA Toolkit のバージョン確認
nvcc --version
```
```bash:nvcc 出力例
$ nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on Thu_Mar_28_02:30:10_Pacific_Daylight_Time_2024
Cuda compilation tools, release 12.4, V12.4.131
Build cuda_12.4.r12.4/compiler.34097967_0
```
最後の行の `Build cuda_` の後に CUDA Toolkit のバージョンが表示されています
これの例では 12.4 であることがわかります
[PyTorch と CUDA Toolkit のバージョンコンパチビリティの確認](https://pytorch.org/get-started/locally/)でインストールコマンドを確認します。
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/74534/6ad00775-7579-ba24-0602-ba394f4901cb.png)

この例の場合のインストールコマンドは次のようになります。
```bash:PyTorchインストールコマンド
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```
FLUX.1 を使用するためには `torch` だけあれば良いので torchvision と torchaudio を除いた以下のコマンドでも大丈夫です
```bash:PyTorchインストール
pip install torch --index-url https://download.pytorch.org/whl/cu124
```
CUDA対応の PyTorch がインストールされたかどうか確認します
```bash:PyTorchバージョン確認
pip show torch
```

#### Hugging Face へログイン（Dev を使用する場合のみ）
- Hugging Face のアカウントを取得済みである必要があります（[Hugging Face アカウント作成ページ](https://huggingface.co/join) で無料で登録することができます）
- Hugging Face の User Access Token を取得している必要があります（[Hugging Face User Access Token 作成ページ](https://huggingface.co/settings/tokens) で無料で取得できます。"Type" は、"READ"）

##### Hugging Face CLI のインストール
```
pip install huggingface_hub
```
##### Hugging Face にログイン
ターミナル（コマンドプロンプト）で次のコマンドを実行します
```
huggingface-cli login
```
Hugging Face のバナーが表示されてトークン（User Access Token）の入力を促されます
```
(.venv) Japanese_Stable_VLM $ huggingface-cli login

    _|    _|  _|    _|    _|_|_|    _|_|_|  _|_|_|  _|      _|    _|_|_|      _|_|_|_|    _|_|      _|_|_|  _|_|_|_|
    _|    _|  _|    _|  _|        _|          _|    _|_|    _|  _|            _|        _|    _|  _|        _|
    _|_|_|_|  _|    _|  _|  _|_|  _|  _|_|    _|    _|  _|  _|  _|  _|_|      _|_|_|    _|_|_|_|  _|        _|_|_|
    _|    _|  _|    _|  _|    _|  _|    _|    _|    _|    _|_|  _|    _|      _|        _|    _|  _|        _|
    _|    _|    _|_|      _|_|_|    _|_|_|  _|_|_|  _|      _|    _|_|_|      _|        _|    _|    _|_|_|  _|_|_|_|

    To login, `huggingface_hub` requires a token generated from https://huggingface.co/settings/tokens .
Token can be pasted using 'Right-Click'.
Enter your token (input will not be visible):
```
ここで Hugging Face のトークン（User Access Token）を入力します
```
Add token as git credential? (Y/n)
```
ここで、トークンを Git の認証情報として追加するかを選択します。どちらでも本アプリケーションの実行には影響ありません
トークンが正しければ次のように表示されてログイン完了です
```
Token is valid (permission: read).
Your token has been saved to C:\Users\yujim\.cache\huggingface\token
Login successful
```
#### 依存パッケージのインストール
```bash:依存パッケージインストール
pip install -r requirements.txt
```
#### Diffusers版サンプルコードの実行
```bash:ヘルプ（オプション）の表示
FLUX1Quantize_and_Generate.py -h
```
```bash:ヘルプの出力
$ python FLUX1Quantize_and_Generate.py -h
usage: FLUX1Quantize_and_Generate.py [-h] [--model {flux-schnell,flux-dev}] [--offload] [--weight {int2,int4,int8,fp8,none}] [--share] [--inbrowser]

Flux.1 with Diffusers demo app

options:
  -h, --help            show this help message and exit
  --model {flux-schnell,flux-dev}
                        Model name(default: flux-schnell)
  --offload             Offload model to CPU when not in use
  --weight {int2,int4,int8,fp8,none}
                        quantization precision(default: int8)
  --share               Create a public link to your demo
  --inbrowser           Launch the demo in the browser
```
モデルには、FLUX.1 Schnell（デフォルト） を int8 量子化（デフォルト）、Model Offload あり、Public URL で共有あり、自動的にブラウザに表示でアプリを起動するには次のコマンドを使います
```bash:アプリ起動
python FLUX1Quantize_and_Generate.py --offload --share --inbrowser
```
起動には、量子化（weight に int2, int4, init8（デフォルト）, fp8 を指定）した場合は 私の環境 で 4分程度、量子化しない場合（weight に none を指定した場合）に 1分程度かかります

アプリが起動すると次のようなメッセージが表示されます
```bash:コンソールメッセージ
...
...
パイプラインの作成時間: 52.15 秒
Running on local URL:  http://127.0.0.1:7860
Running on public URL: https://xxxxxxxx.gradio.live

This share link expires in 72 hours. For free permanent hosting and GPU upgrades, run `gradio deploy` from Terminal to deploy to Spaces (https://huggingface.co/spaces)
```
ブラウザが自動的に立ち上がらない場合は、ローカル環境であれば local URL の http://127.0.0.1:7860 を、クラウドなどのリモート環境であれば public URL の https://xxxxxxxx.gradio.live をブラウザで開きます（xxxxxxxx は都度異なります）。この public URL は アプリを終了しない限り 72時間有効です。アプリを終了する際は Ctrl-C で終了します

##### 画像生成
<table>
<tr>
<td>

![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/74534/b7152bec-cf48-97d4-79f5-20a64979828a.png)
</td>
</tr>
<tr>
<td>
量子化しない（weight = none）場合の表示例
</td>
</tr>
</table>

詳細設定をクリックすると画像のサイズなどを設定できます

![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/74534/eca13a83-248c-8479-3cc8-c8a1189c8904.png)

左上の "Prompt" に生成したい画像のプロンプト（英語）を入力して「画像生成」ボタンをクリックすると生成が始まります


##### 生成完了時の画面
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/74534/2c0384cd-196e-d568-e57b-4a4a06b3952b.png)

##### 生成画像例
<table>
<tr>
<td>

![flux-schnell-output2.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/74534/e2a59a47-41d0-375e-d607-72d5a545d3fb.png)
</td>
<td>

![flux-schnell-output.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/74534/79a3459d-f08b-5875-fe3f-9455c2995466.png)

</td>
</tr>
<tr>
<td>
A cat holding a sign that says Hello Kitty
</td>
<td>
A cute anime girl is standing on the hill. She has black hair, a bob, a white beret, and a red cardigan over a white blouse. Her skirt is a white fluffy mini frill skirt with a large spread. She is wearing white knee-high socks. Her shoes are pink. Cherry blossoms are dancing on the asphalt slope leading to the hill. The composition is looking up at her from the bottom of the hill
</td>
</tr>
<tr>
<td>

![aek3s-wuk7r.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/74534/2d3a4fa6-60f0-1da8-908e-0b284bbd3396.png)

</td>
<td>

![a079f-y2om1.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/74534/c5716627-1996-2433-9a9c-fc0b6b6944c5.png)
</td>
</tr>
<tr>
<td>
3D animation style graphics reminiscent of Pixar of a magical girl in a pink costume. She is holding a magic wand. Beside her is a white cat
</td>
<td>
Cat driving a motorcycle
</td>
</tr>
</table>

#### サンプルコードの説明
```python:Diffusers FluxPipeline のインポート
from diffusers import FluxPipeline
```
FLUX.1 でテキストから画像を生成する pipeline のクラスです
```python:Diffusers Pipeline の生成
def create_pipe(model: str, offload = True, weight = "int8"):
    if model == "flux-schnell":
        bfl_repo = "black-forest-labs/FLUX.1-schnell"
    elif model == "flux-dev":
        bfl_repo = "black-forest-labs/FLUX.1-dev"
    else:
        raise ValueError(f"Invalid model name: {model}")

    start_time = time.time()
    pipe = FluxPipeline.from_pretrained(
        bfl_repo, torch_dtype=torch.float16
    )

    if weight != "none":
        quantize(pipe.transformer, weights=UNET_QTYPES[weight], exclude=["proj_out", "x_embedder", "norm_out", "context_embedder"])
        freeze(pipe.transformer)
        quantize(pipe.text_encoder_2, weights=UNET_QTYPES[weight])
        freeze(pipe.text_encoder_2)

    if offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"パイプラインの作成時間: {execution_time:.2f} 秒")
    return pipe
```

`FluxPipeline.from_pretrained(bfl_repo, torch_dtype=torch.float16)` で `bfl_repo` に指定された Hugging Face Hub のリポジトリからモデルをダウンロードして、text2image の pipeline を生成しています

起動パラメータの weight に none 以外が指定された場合、Optimum Quanto の `quantize()` を使って量子化してます

また、起動パラメータ offload が指定された場合は、`enable_model_cpu_offload()`で Model Offloading しています。これは、モデルのコンポーネントが必要なときにだけ GPU VRAM に載り、使用していないときには CPU側のメモリへ退避することで GPU VRAM の必要量を削減する機能です

## おまけ

他にもいろいろ記事を書いていますので良かったらお立ち寄りください。

https://qiita.com/yuji-arakawa/items/1a8cfeff8f81ba808389

https://qiita.com/yuji-arakawa/items/9cd485debd5b0d18aca2

https://qiita.com/yuji-arakawa/items/042937eaf16fa00cf491

https://qiita.com/yuji-arakawa/items/6d0299c505315bc3cdb0

https://qiita.com/yuji-arakawa/items/05e3455572d3b09a53dc

https://qiita.com/yuji-arakawa/items/2d4f6eff17a5410dba2d

https://qiita.com/yuji-arakawa/items/597c4bd9f3d5b4212b51

https://aws.amazon.com/jp/blogs/news/leveraging-pinecone-on-aws-marketplace-as-a-knowledge-base-for-amazon-bedrock/