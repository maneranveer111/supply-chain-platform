from app.workers.celery_app import celery_app


@celery_app.task(bind=True, max_retries=3)
def retrain_model(self):
    """
    Kicks off LSTM/Prophet retraining. In practice, this would call into
    a training module built from the notebook pipeline (app/ml/train.py,
    not included in the scaffold) and write new artifacts to saved_models/.
    """
    try:
        # TODO: call the actual training pipeline
        print("Retraining triggered (placeholder — wire up app.ml training pipeline)")
    except Exception as exc:
        raise self.retry(exc=exc, countdown=60 * (2**self.request.retries))
