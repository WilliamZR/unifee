from torchnet import meter

class EvalMetric(object):
    def __init__(self):
        self.loss_meter = meter.AverageValueMeter()
        self.acc_meter = meter.AverageValueMeter()
        self.rec_meter = meter.AverageValueMeter()
        self.prec_meter = meter.AverageValueMeter()

    def meter_reset(self):
        self.loss_meter.reset()
        self.acc_meter.reset()
        self.rec_meter.reset()
        self.prec_meter.reset()

    def meter_add(self, acc, recall, prec, loss):
        self.loss_meter.add(loss)
        self.acc_meter.add(acc)
        self.rec_meter.add(recall)
        self.prec_meter.add(prec)

    def print_meter(self):
        print("loss:", self.loss_meter.value()[0])
        print("accuracy:", self.acc_meter.value()[0])
        print("recall:", self.rec_meter.value()[0])
        print("precision:", self.prec_meter.value()[0])
        print('\n')

