
# An-assistant-to-the-basic-principles-of-Marxism

## 环境的配置
使用pip安装requirements中所需要的包（conda可能出现不兼容的问题），同时pyhon版本建议在3.10以上
## 知识库的构建
- 先将所需训练的原始材料放入mayuan_raw_data文件夹
- 运行genetrate_database，会自动将知识库构建在database_agent_mayuan这个文件夹中，后续更新知识库的时候再次运行即可
## 出题模型调用
运行mayuan_agent即可调用模型，同样模型的整体架构搭建也在这个脚本中，可通过修改架构实现不同的功能，在终端中输入quit即可退出模型
## 知识图谱模型的调用
运行mayuan_kg_agent即可调用知识图谱模型，但在终端调用时仅能返回mermaid代码，若想看完整思维导图可以在网站段查看
## 函数的封装
可在不同主题模型如毛概、习概等可迁移的函数均封装在common_utils这个文件夹下面
### base_agent+prompts
有关出题agent的整体结构函数以及提示词
### base_kg_agent
有关思维图谱agent的整体结构函数
### llm_wrapper
与模型api接口有关的函数
### vector_utils
向量处理有关函数
## 网站的调用
运行app.py文件根据给出的链接则可呈现网站
