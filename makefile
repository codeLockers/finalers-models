# =============================================================================
#  Makefile 示例 —— 在 Makefile 里「定义命令」并复用
#
#  用 `make` 或 `make help` 列出全部命令，例如：
#      make check-env                     # 环境自检
#      make split-data SRC=full.txt       # 按标签拆成 data/labeling/<标签>_data.txt
#                                         #   留出集同理: make split-data SRC=old_holdout.txt LABEL_DIR=data/probe
#      make dry-run                       # 只检查数据，不训练
#      make train                         # 训练 + 留出集评估 -> models/intent.bin
#      make predict TEXT="午饭花了30块"    # 用 .bin 模型预测
#      make predict-ftz TEXT="午饭花了30块" # 用 .ftz 端侧模型预测
#      make eval-ftz                      # 用留出集测试 .ftz 模型
#      make tune DURATION=120             # 自动调参
#      make mobile                        # 复用 .bin 量化导出 -> models/finalers_mobile.ftz
#      make mobile-train                  # 一步训练+量化端侧模型（dim=20, bucket=50000）
#      make clean                         # 清理产出
#
#  两个必须知道的规则：
#    1) recipe 的每一行都必须以 TAB 开头（不能用空格），否则报
#       "missing separator"；
#    2) `make -n <target>` 只打印将要执行的命令、不真正执行，用来检查变量展开。
# =============================================================================

# -----------------------------------------------------------------------------
# 1) 变量：把路径 / 参数 / 整条命令抽出来，改一处即可全局生效
#
#    ?=  若命令行或环境已定义则不覆盖：make train PYTHON=python3.11
#    =   递归展开（真正用到时才求值）
#    :=  立即展开（定义时求值，配合 $(shell ...) 用）
# -----------------------------------------------------------------------------
PYTHON    ?= python3
SCRIPT    ?= fasttext_training_script.py

DATA      ?= data/labeling
# 留出集：可以是单个文件，也可以是用 --split-by-label 拆出来的目录（目录内 *.txt 会被合并）
HOLDOUT   ?= data/probe
# 按标签拆分数据的输出目录（--split-by-label 用；也可以直接当作 -i 的输入目录）
LABEL_DIR ?= data/labeling
MODEL_DIR ?= models
MODEL     ?= $(MODEL_DIR)/finalers.bin
MOBILE    ?= $(MODEL_DIR)/finalers_mobile.ftz
THREAD    ?= 4
DURATION  ?= 60
TEXT      ?= 午饭花了30块
# 端侧小模型的训练超参（mobile-train 用；bucket 不要低于 50000，否则易 NaN）
MOBILE_DIM    ?= 20
MOBILE_BUCKET ?= 50000

# 「一条完整命令」也可以直接定义成变量；recipe 里写 $(TRAIN) 就是展开整行
PY       := $(PYTHON) $(SCRIPT)
TRAIN     = $(PY) -i $(DATA) -o $(MODEL) -t $(HOLDOUT) --thread $(THREAD)
TUNE      = $(PY) -i $(DATA) -o $(MODEL_DIR)/tuned.bin --autotune --autotune-duration $(DURATION)
QUANTIZE  = $(PY) -m $(MODEL) --quantize -o $(MOBILE) -t $(HOLDOUT)
# 一步到位：直接用原始数据训练小模型并量化，产物同样是 $(MOBILE)
MOBILE_TRAIN = $(PY) -i $(DATA) -o $(MOBILE) -t $(HOLDOUT) \
               --dim $(MOBILE_DIM) --bucket $(MOBILE_BUCKET) --quantize

# DATA/HOLDOUT 是目录时，依赖必须落到具体文件上：只依赖目录的话，加减文件才会更新目录 mtime，
# 原地改写 data/labeling/*.txt 不会触发重训（make 只比较 mtime）。
# 目录 -> 该目录下所有 *.txt；单文件 -> 该文件本身。$(call 函数名,参数) 调用自定义函数。
files_of = $(if $(wildcard $(1)/*.txt),$(wildcard $(1)/*.txt),$(1))
DATA_FILES    := $(call files_of,$(DATA))
HOLDOUT_FILES := $(call files_of,$(HOLDOUT))

# $(shell ...) 在解析阶段执行一次，结果缓存在变量里（不是每条命令都跑）
GIT_REV  := $(shell git rev-parse --short HEAD 2>/dev/null || echo unknown)

.DEFAULT_GOAL := help     # 只敲 make 时默认执行 help
.DELETE_ON_ERROR:         # 命令失败时删掉写了一半的目标文件

# -----------------------------------------------------------------------------
# 2) 自文档化 help：行尾 `## 说明` 会被 grep 收集，新增 target 不用改这里
# -----------------------------------------------------------------------------
help: ## 显示可用命令
	@echo "用法: make <target> [变量=值]"
	@echo "例:   make train MODEL=$(MODEL_DIR)/demo.bin THREAD=8"
	@echo
	@grep -hE '^[a-zA-Z0-9_.-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'
	@echo
	@echo "当前: PYTHON=$(PYTHON)  DATA=$(DATA)  MODEL=$(MODEL)  rev=$(GIT_REV)"

# -----------------------------------------------------------------------------
# 3) 普通命令：每条 recipe 都要 TAB 开头
#      @  不打印命令本身，只打印输出
#      -  该行失败也继续执行
# -----------------------------------------------------------------------------
install: ## 安装依赖（README 推荐 --user）
	@echo ">> 安装 requirements.txt"
	$(PYTHON) -m pip install --user -r requirements.txt

check-env: ## 环境自检：用内置小样例真的训练 + 预测一次
	@echo ">> Python: $$($(PYTHON) --version 2>&1)"
	$(PY) --check-env

dry-run: ## 只做数据检查与切分，不训练
	$(PY) -i $(DATA) --dry-run

split-data: ## 按标签拆分完整数据：SRC=源文件（输出到 LABEL_DIR，每类一个文件）
	@test -n "$(SRC)" || { echo "用法: make split-data SRC=/path/to/full.txt [LABEL_DIR=data/labeling]"; exit 2; }
	@mkdir -p $(LABEL_DIR)
	@echo ">> 按标签拆分 $(SRC) -> $(LABEL_DIR)/<标签>_data.txt"
	$(PY) -i $(SRC) --split-by-label $(LABEL_DIR)

print-vars: $(DATA_FILES) ## 打印变量展开结果（调试 Makefile 用）
	@echo "MODEL           = $(MODEL)"
	@echo "DATA_FILES      = $(DATA_FILES)"
	@echo "HOLDOUT_FILES   = $(HOLDOUT_FILES)"
	@echo "TRAIN           = $(TRAIN)"
	@echo "QUANTIZE        = $(QUANTIZE)"
	@echo '$$@ 目标        = $@'
	@echo '$$< 首个依赖    = $<'
	@echo '$$^ 全部依赖    = $^'
	@echo '$$(@D) 目标目录 = $(@D)   # 这里目标就是 print-vars, 所在目录是 .'

# -----------------------------------------------------------------------------
# 4) 文件目标：只有数据比模型新才会重新训练
#    `train` 只是模型的别名，所以连敲两次 make train 第二次会跳过
# -----------------------------------------------------------------------------
$(MODEL): $(DATA_FILES)
	@mkdir -p $(@D)
	@echo ">> 训练 $(MODEL) <- $(DATA) (rev=$(GIT_REV))"
	$(TRAIN)

train: $(MODEL) ## 训练并用留出集评估（已是最新则跳过）

force-train: ## 无条件重训（忽略文件时间戳）
	@mkdir -p $(MODEL_DIR)
	@echo ">> 强制重训 $(MODEL)"
	$(TRAIN)

tune: ## 自动调参（DURATION=秒 控制时长，默认 60）
	@mkdir -p $(MODEL_DIR)
	$(TUNE)

# 模式规则：一条命令适配所有 models/xxx.bin（make models/foo.bin 会用 data/foo.txt 训练）
$(MODEL_DIR)/%.bin: data/%.txt
	@mkdir -p $(@D)
	$(PY) -i $< -o $@ --thread $(THREAD)

# -----------------------------------------------------------------------------
# 5) 依赖驱动的量化导出：.ftz 依赖 .bin，.bin 更新了才会重新量化
# -----------------------------------------------------------------------------
mobile: $(MOBILE) ## 复用已有 .bin 量化导出端侧模型 models/finalers_mobile.ftz

$(MOBILE): $(MODEL) $(HOLDOUT_FILES)
	@mkdir -p $(@D)
	$(QUANTIZE)

mobile-train: ## 从原始数据一步训练+量化端侧模型（dim=20, bucket=50000）
	@mkdir -p $(MODEL_DIR)
	@echo ">> 训练+量化 $(MOBILE)  (dim=$(MOBILE_DIM), bucket=$(MOBILE_BUCKET))"
	$(MOBILE_TRAIN)

# -----------------------------------------------------------------------------
# 6) 推理相关
# -----------------------------------------------------------------------------
predict: ## 用 .bin 模型单条文本预测，TEXT="..." 传句子
	$(PY) -m $(MODEL) --predict "$(TEXT)"

predict-ftz: ## 用 .ftz 端侧模型单条文本预测（需先 make mobile-train）make predict-ftz TEXT="午饭花了30块"
	$(PY) -m $(MOBILE) --predict "$(TEXT)"

eval-ftz: ## 用留出集测试 .ftz 模型，输出条目数 / 成功率 / 失败明细（--metrics 看 P/R/F1）
	$(PY) -m $(MOBILE) -t $(HOLDOUT)

predict-file: $(HOLDOUT_FILES) ## 对留出集逐行预测，输出 JSONL（HOLDOUT 可以是目录）
	@mkdir -p $(MODEL_DIR)
	$(PY) -m $(MODEL) --predict-file $(HOLDOUT) > $(MODEL_DIR)/preds.jsonl
	@echo ">> 结果写入 $(MODEL_DIR)/preds.jsonl"

interactive: ## 交互式预测（Ctrl-D 退出）
	$(PY) -m $(MODEL) -I

report: $(MODEL) ## 查看训练报告 JSON
	@$(PYTHON) -m json.tool $(MODEL).report.json

# -----------------------------------------------------------------------------
# 7) 清理
# -----------------------------------------------------------------------------
clean: ## 删除模型 / 报告 / 预测结果
	@echo ">> 清理 $(MODEL_DIR)/"
	-rm -rf $(MODEL_DIR)/*.bin $(MODEL_DIR)/*.ftz $(MODEL_DIR)/*.json $(MODEL_DIR)/preds.jsonl

distclean: clean ## 连临时文件、Python 缓存一起删
	-rm -rf work __pycache__ .pytest_cache
	-find . -name '*.pyc' -delete

# 声明「假目标」：同名文件存在也不会跳过，且不参与依赖时间戳比较
.PHONY: help install check-env dry-run split-data print-vars train force-train tune mobile \
        mobile-train a predict predict-ftz eval-ftz predict-file interactive \
        report clean distclean
