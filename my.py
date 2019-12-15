from ai import from_sensor_code as code

sensor = code.MyClassifier();
sensor.read('sleepdata_v2.csv')
sensor.show()
sensor.plot('time', 'sleep_time', 'sleep')
sensor.histogram()
sensor.violinplot('time', 'sleep')
sensor.violinplot('sleep_time', 'sleep')
sensor.heatmap()
sensor.run_logistic_regression(["time", "sleep_time"], "sleep")
sensor.run_decision_tree_classifier(["time", "sleep_time"], "sleep")
sensor.run_svm(["time", "sleep_time"], "sleep")
for i in range(8):
    a = i+1
    print(a)
    sensor.run_neighbor_classifier(["time", "sleep_time"], "sleep", a)
sensor.run_all(["time", "sleep_time"], "sleep", 4)
sensor.draw_4_accuracy()