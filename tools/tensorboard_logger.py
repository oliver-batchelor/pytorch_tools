import tensorflow as tf

class TensorboardLogger:
    def __init__(self, log_dir):
        self.writers = {None : tf.summary.FileWriter(log_dir)}
        self.log_dir = log_dir

    def flush(self):
        for writer in self.writers.values():
            writer.flush()

    def scalar(self, tag, value, step):
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag,  simple_value=value)])
        self.writer(run=run).add_summary(summary, step)

    def scalars(self, tag, value_dict, step):
        for run, value in value_dict.items():
            summary = tf.Summary(value=[tf.Summary.Value(tag=tag,  simple_value=value)])
            self.writer(run=run).add_summary(summary, step)

                    
    def pr_curve(self, tag, curve, step):
        pass

        # summary_metadata = metadata.create_summary_metadata(display_name=tag, description='', num_thresholds=curve.precision.size(0))
        # summary = tf.Summary()

        # data = torch.stack([curve.true_positives, curve.False_positives, 
        #     curve.true_negatives, curve.False_negatives, 
        #     curve.precision, curve.recall])

        # tensor = tf.make_tensor_proto(np.float32(data.numpy()), dtype=tf.float32)
        # summary.value.add(tag=tag, metadata=summary_metadata, tensor=tensor)         

        # self.add_summary(summary, run=run)


    def histogram(self, tag, histogram, step):
        assert isinstance(histogram, Histogram)
        # """Logs the histogram of a list/vector of values."""

        # # Fill fields of histogram proto
        hist = tf.HistogramProto()
        hist.min = histogram.range[0]
        hist.max = histogram.range[1]
        hist.num = histogram.counts.sum().item()
        
        hist.sum = histogram.sum
        hist.sum_squares = histogram.sum_squares

        # Requires equal number as bins, where the first goes from -DBL_MAX to bin_edges[1]
        # See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/summary.proto#L30
        # Thus, we drop the start of the first bin

        #Add bin edges and counts
        bin_edges = histogram.bins()[1:]

        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in histogram.counts:
            hist.bucket.append(c)

        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.add_summary(summary, step)


    # Internal
    def writer(self, run = None):       
        if not (run in self.writers):
            self.writers[run] = tf.summary.FileWriter(path.join(self.log_dir, str(run)))

        return self.writers[run]

    def add_summary(self, summary, step, run = None):
        self.writer(run).add_summary(summary, global_step = step)
