# train.py
import tensorflow as tf
from model import Transformer
from data_load import get_batch
from hparams import Hparams
import os
import logging

logging.basicConfig(level=logging.INFO)

def main():
    # 1. 加载超参数
    logging.info("# hparams")
    hparams = Hparams()
    parser = hparams.parser
    hp = parser.parse_args()

    # 2. 准备训练和验证数据
    logging.info("# Prepare train/eval batches")
    train_batches, num_train_batches, _ = get_batch(hp.train1, hp.train2,
                                                    hp.maxlen1, hp.maxlen2,
                                                    hp.vocab, hp.batch_size,
                                                    shuffle=True)
    eval_batches, num_eval_batches, _ = get_batch(hp.eval1, hp.eval2,
                                                  100000, 100000,
                                                  hp.vocab, hp.batch_size,
                                                  shuffle=False)
    
    # 3. 创建数据迭代器
    iter = tf.data.Iterator.from_structure(train_batches.output_types, train_batches.output_shapes)
    xs, ys = iter.get_next()

    train_init_op = iter.make_initializer(train_batches)
    eval_init_op = iter.make_initializer(eval_batches)

    # 4. 加载模型
    logging.info("# Load model")
    model = Transformer(hp)
    loss, train_op, global_step, train_summaries = model.train(xs, ys)

    # 5. 开始训练
    saver = tf.train.Saver(max_to_keep=hp.num_epochs)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        summary_writer = tf.summary.FileWriter(hp.logdir, sess.graph)

        sess.run(train_init_op)  # 初始化迭代器
        total_steps = hp.num_epochs * num_train_batches

        for step in range(total_steps):
            _, gs, summary = sess.run([train_op, global_step, train_summaries])
            summary_writer.add_summary(summary, gs)

            # 每完成一个 epoch，执行验证
            if gs % num_train_batches == 0:
                logging.info("Epoch {} done".format(gs // num_train_batches))
                sess.run(eval_init_op)  # 切换到验证集
                sess.run([loss])  # 计算验证损失

        summary_writer.close()
        saver.save(sess, os.path.join(hp.logdir, 'model.ckpt'))

if __name__ == '__main__':
    main()
