from markitdown import MarkItDown

# 初始化转换器
md = MarkItDown()

# 无论是 pptx, xlsx 还是 docx，都是同一个用法
result = md.convert("C:\\Users\\Shengyi LIU\\Downloads\\青蒿素.docx")

# 输出转换后的 Markdown 文本
print(result.text_content)