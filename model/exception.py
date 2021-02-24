class WeightsInitializeError(Exception):
    """
    自定义权重初始化异常.
    """
    def __init__(self, message):
        super(WeightsInitializeError, self).__init__(message)
        self.message = message
