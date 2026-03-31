#!/bin/bash
set -e
set -x
bcecmd=/Users/yexianbin/mac-bcecmd-0.5.4/bcecmd
python_bin=/usr/bin/python3 ## should be installed with paddle

CKPT_BOS_PATH=$1        # 指定一个 BOS 下的 ckp 路径，自动下载下来部署。比如：bos:/helixvs-temp/backdoor/r012_698k_online.pdparams
EXTRACT_PARAM=$2        # 是否需要从 ckp 中提取 'model' 字段下的参数，默认应该不需要，假定模型侧已经处理好；若需要，则需要 python & paddle

UPLOAD_BOS=1            # 1 为上传，其他值为不上传
RUN_TEST=""             # 触发测试的级别，具体见 pytest.ini。设为空则不触发测试

SCRIPT_DIR=$(cd `dirname $0`; pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../" && pwd)
PROJECT_NAME=`basename $PROJECT_ROOT`

## Download ckpt from bos and compress it, finally update `INIT_MODEL` in chpc_infer.sh
if [[ -n ${CKPT_BOS_PATH} ]]; then
    echo "[NOTE] Update ckpt from ${CKPT_BOS_PATH}"
    ckpt_file=$(basename ${CKPT_BOS_PATH})
    ckp_deploy_path=${PROJECT_ROOT}/data/infer_online/params/

    if [[ ! -f ${ckp_deploy_path}/${ckpt_file} ]]; then
        cd "${SOURCE_DIR}"
        rm -rf "$ckp_deploy_path"
        mkdir -p "$ckp_deploy_path"
        $bcecmd bos cp ${CKPT_BOS_PATH} "$ckp_deploy_path/"
        if [[ -z "$ckpt_file" ]]; then
            echo "Error: No checkpoint file found. Exiting."
            exit 1
        fi
        echo "ckpt_file: $ckpt_file"
        if [[ -n ${EXTRACT_PARAM} ]]; then
            if [[ ! -e "${python_bin}" ]]; then
                echo "Error: python not found. Exiting."
                exit 1
            fi
            ${python_bin} scripts/ckpt_compress.py "$ckp_deploy_path/$ckpt_file"
            if [[ $? -ne 0 ]]; then
                echo "Error: ckpt_compress.py failed."
                exit 1
            fi
        fi
        md5sum "$ckp_deploy_path/$ckpt_file" > "${PROJECT_ROOT}/infer_scripts/ckpt_md5"
        sed -i "s|^INIT_MODEL=.*|INIT_MODEL=\"\$PROJECT_ROOT/data/infer_online/params/$ckpt_file\"|" "scripts/chpc_infer.sh"
        echo "Push modified chpc_infer.sh and call this script again. Exit !!!"
        exit 1
        cd ../ 
    fi
fi

## Sync source code and ckp to a new directory
SOURCE_DIR="$PROJECT_NAME"
SYNC_DIR="${SOURCE_DIR}_sync"

cd $PROJECT_ROOT/../
rm -rf $SYNC_DIR
# Set COPYFILE_DISABLE to prevent creation of ._* files on macOS
export COPYFILE_DISABLE=1
rsync -a --exclude '.*/' --exclude '._*' --include '*/' \
    --include '*.py' --include '*.sh' --include '*.md' --include '*.json' --include '*.json-eval' --include '*.ini' \
    --include 'libstdc*' --include 'ccd_database.json.gz' \
    --include 'infer_scripts/testcases/ref_structure/*.cif' \
    --include 'infer_scripts/testcases/s1/*.cif' \
    --exclude '*' \
    $SOURCE_DIR/ \
    $SYNC_DIR/
if [[ -n ${CKPT_BOS_PATH} ]]; then
    cp -r $SOURCE_DIR/data/infer_online/params $SYNC_DIR/data/infer_online/
    cp $SOURCE_DIR/infer_scripts/ckpt_md5 $SYNC_DIR/infer_scripts/ckpt_md5
fi

# ## Pack test cases for easier sync
# rsync -a --exclude '*eval' $SOURCE_DIR/infer_scripts/testcases/* $SOURCE_DIR/infer_scripts/online_testcase/
# tar -C $SOURCE_DIR/infer_scripts/ -cf $SOURCE_DIR/infer_scripts/online_testcase.tar online_testcase
# rm -rf $SOURCE_DIR/infer_scripts/online_testcase

## Pack version info
cd "$SOURCE_DIR" 
commit_id=$(git rev-parse HEAD)
tag_id=$(git describe --tags --exact-match HEAD 2>/dev/null || echo "helixfold3")
branch=$(git rev-parse --abbrev-ref HEAD)
echo "$commit_id" > "$(pwd)/../$SYNC_DIR/infer_scripts/commit_id"
echo "$tag_id" >> "$(pwd)/../$SYNC_DIR/infer_scripts/commit_id"
echo "$branch" >> "$(pwd)/../$SYNC_DIR/infer_scripts/commit_id"
cd ../

rm -f protein_folding.tar
if [[ "$(uname)" == "Darwin" ]]; then
    tar --no-xattrs --no-mac-metadata -cf protein_folding.tar $SYNC_DIR
else
    tar --no-xattrs -cf protein_folding.tar $SYNC_DIR
fi

## Upload to bos
if [[ ${UPLOAD_BOS} -eq 1 ]]; then
    $bcecmd bos cp protein_folding.tar bos:/helixvs-temp/backdoor/

    set +x
    # 注意：MSA 动态缓存路径的索引在网络磁盘：/data/helixfold/msa_cache/ind/，自动调整路径权限保证访问
    echo "Done. Run under your deploy path on CHPC master node:"
    echo "rm -rf infer_scripts/testcases/ && bcecmd bos cp bos:/helixvs-temp/backdoor/protein_folding.tar ./ && tar -xvf protein_folding.tar --strip-components=1 --no-same-owner && bash scripts/chpc_deploy.sh \"${RUN_TEST}\""
else
    set +x
    echo "Do NOT upload to BOS. Done."
fi
