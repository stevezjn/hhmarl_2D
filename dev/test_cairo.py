import cairo

# 创建 PDF 表面
surface = cairo.PDFSurface("output.pdf", 400, 300)
ctx = cairo.Context(surface)

# 绘制红色矩形
ctx.set_source_rgb(1, 0, 0)
ctx.rectangle(50, 50, 200, 100)
ctx.fill()

surface.finish()  # 保存文件