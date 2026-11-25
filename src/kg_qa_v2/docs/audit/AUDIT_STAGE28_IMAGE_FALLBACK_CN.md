# Stage 28 审计：图片证据与来源链接兜底

## 目标

修复A端图片证据显示异常、代理图片地址加载失败后出现破损图片，以及点击来源链接打开空白/登录跳转的问题。

## 原因

B端图谱证据同时返回：
- A端图片代理地址：/zhangbi_feedback/wiki_custom/kg-qa/assets/793/
- 原始媒体地址：https://ai.smartoptiks.cn/zhangbi_feedback/media/assets/images/2026/05/c969b58f26b2440d80779ac5bc50c2a2.webp

代理地址受登录态和当前领域校验影响。实际未带A端登录会话访问时返回 302 登录跳转；原始媒体地址返回 200。因此只使用代理地址不够稳健。

## 修改文件

- `wiki_custom/templates/wiki_custom/kg_qa_widget.html`

## 修改内容

1. 证据图片同时记录代理地址和资产的原始媒体地址。
2. 图片加载失败时，自动切换到原始媒体地址。
3. 两个地址都失败时移除图片元素，避免页面显示破损图片图标。
4. “查看来源”对图片证据优先使用资产原始媒体地址；普通文本证据仍使用原有 source_url。
5. 答案图片同样支持 source_url 兜底。

## 输入输出审计

输入：
- B端 evidence.metadata.assets 中包含 image_url 和 source_url。

输出：
- 首选 image_url，保持现有领域权限代理路径。
- 代理失败后切换 source_url。
- source_url 与 image_url 相同或为空时，不重复切换。

## 测试

- A端 Django 流式与V2集成测试：7 passed。
- A端模板脚本提取后执行 `node --check)：通过。
- B端真实查询：HTTP 200，返回 5 条图谱证据，并返回原始图片来源地址。
- 代理地址行为核验：未带登录会话时返回 302；原始媒体地址返回 200、40116 bytes。
