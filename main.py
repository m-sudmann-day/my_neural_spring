from model import *
from history import *
from misc import *

notes = "128x128, Keras, 100 files"

# Run this first because it starts logging stdout and stderr to files.
history = history()
history.start()
history.write_text_file("notes.txt", notes)
history.copy_files_to_new_folder("*", "scripts")
history.copy_files_to_new_folder("../images/*", "images")
print_versions()
history.stop()
quit()

net = model(seed=12345, history_path=history.output_path)

net.load_data(train_ratio=0.75)

net.train(num_epochs=3, batch_size=128)

train_result = net.test(use_training_set=True)
test_result = net.test()

print("Train accuracy: %.2f%%" % (train_result * 100))
print("Test accuracy: %.2f%%" % (test_result * 100))

net.predict()

history.write_stub(str(test_result * 100) + ".result")
history.stop()

print("done")

# Workaround for an intermittent error that pops up in session.py and other places
del net
# There is still another intermittent TensorFlow cleanup error without a workaround.