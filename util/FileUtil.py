def readAllLines(path: str) -> []:
    """
    一次性读取全部行,删除每行前后的空格
    :param path: 文件路径
    :return: 由每一行组成的list
    """
    allLines = []
    with open(path, 'r') as f:
        for line in f.readlines():
            allLines.append(line.strip())
    return allLines
