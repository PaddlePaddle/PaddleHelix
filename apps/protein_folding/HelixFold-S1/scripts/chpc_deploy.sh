#! /bin/bash
set -e
set -x

RUN_TEST=$1

if [[ $EUID -eq 0 ]]; then
    chmod -R 777 /data/helixfold/msa_cache/ind/
fi

# 下载 ccd 库
if [[ ! -f infer_scripts/demo_data/ccd_preprocessed_etkdg.pkl.gz ]]; then
    /root/linux-bcecmd-0.5.1/bcecmd bos cp -y bos:/helixvs-temp/backdoor/ccd_preprocessed_etkdg.pkl.gz infer_scripts/demo_data/
fi
# 补充 ppfleetx 的环境依赖
if [[ ! -d ppfleetx ]]; then
    /root/linux-bcecmd-0.5.1/bcecmd bos cp -y bos:/helixvs-temp/backdoor/ppfleetx.tar ./
    tar -xf ppfleetx.tar
fi

# # 修改 backdoor 中的路径指向
# cur=`pwd`
# sed -i "s|^PROJECT_ROOT=.*|PROJECT_ROOT=${cur}|" "scripts/chpc_backdoor_s1.sh"

# # 修改参考结构用例中的 cif 路径
# for json_file in infer_scripts/testcases/ref_structure/*.json infer_scripts/testcases/s1/*.json; do
#     if [ -f "$json_file" ]; then
#         # Process each ref_file entry separately
#         while IFS= read -r ref_file; do
#             if [ -n "$ref_file" ]; then
#                 # Check if current path is absolute or file exists
#                 if [[ "$ref_file" != /* && ! -f "$ref_file" ]]; then
#                     # Escape special characters in the path for sed
#                     escaped_path=$(echo "$cur/infer_scripts/testcases/ref_structure/$ref_file" | sed 's/[\/&]/\\&/g')
#                     sed -i "s|\"ref_file\": *\"$ref_file\"|\"ref_file\": \"$escaped_path\"|" "$json_file"
#                 fi
#             fi
#         done < <(grep -o '"ref_file": *"[^"]*"' "$json_file" | cut -d'"' -f4)
#     fi
# done

# test () {
#     # --collect-only
#     /data/helixfold/envs/helixfold3/bin/python -m pytest \
#         --import-mode=importlib -m "$1" infer_scripts/main_test.py -v
# }

# if [[  -z ${RUN_TEST} ]]; then
#     echo "Skip testing."
# else
#     test "${RUN_TEST}"
# fi