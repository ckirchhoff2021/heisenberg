# 任务：数据预览网站
- 实现一个数据预览的网页服务，用于展示图像数据和对应的标签信息


## 核心要求
- 能通过服务化的方式访问数据预览网站
- 能在网站上查看预览每一条图像数据，图像数据下方有它的标签信息，标签信息主要包括`curve`和`recover`两个字段
- 能在网站上进行数据的筛选，筛选条件包括`curve`和`recover`两个字段
- 网站尽可能美观
- 以JSON文件作为输入，后续支持其他数据预览
- 测试可使用`datas/aiops/files/jsons/argos-trains.json`和`datas/aiops/files/jsons/argos-valids.json`两个文件
- 代码存放在`datas/aiops/files`，单独新建`viewer`目录


## 注意事项
- JSON文件中的`image`字段只是图像名称，根据`from`字段确定图像的来源目录，从`datas/aiops/files/gen`和`datas/aiops/files/labeled`目录中读取图像文件

