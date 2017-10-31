from flask import Flask, request

def start_server(target_class, feedback_calc_trial, clf_outs):
    app = Flask(__name__)
    app.target_class = target_class
    app.feedback_calc_trial = feedback_calc_trial
    app.clf_outs = clf_outs
    @app.route('/rt_data', methods=['POST'])
    def rt_data():
        data = request.get_json(force=True)
        app.target_class.value = data['target_class']
        app.clf_outs[:] = data['clf_outs'][:]
        app.feedback_calc_trial.value = data['trial_num']
        return 'data_received'
    @app.route('/shutdown', methods=['POST'])
    def shutdown():
        shutdown_func = request.environ.get('werkzeug.server.shutdown')
        shutdown_func()
        return 'Server shutting down...' 
    app.run()

