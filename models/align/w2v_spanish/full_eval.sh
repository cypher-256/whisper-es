# CV - TEST

python eval.py --model_id jonatasgrosman/wav2vec2-large-xlsr-53-spanish --dataset mozilla-foundation/common_voice_6_0 --config es --split test --log_outputs --greedy
mv log_mozilla-foundation_common_voice_6_0_es_test_predictions.txt log_mozilla-foundation_common_voice_6_0_es_test_predictions_greedy.txt
mv mozilla-foundation_common_voice_6_0_es_test_eval_results.txt mozilla-foundation_common_voice_6_0_es_test_eval_results_greedy.txt

python eval.py --model_id jonatasgrosman/wav2vec2-large-xlsr-53-spanish --dataset mozilla-foundation/common_voice_6_0 --config es --split test --log_outputs

# HF EVENT - DEV

python eval.py --model_id jonatasgrosman/wav2vec2-large-xlsr-53-spanish --dataset speech-recognition-community-v2/dev_data --config es --split validation --chunk_length_s 5.0 --stride_length_s 1.0 --log_outputs --greedy
mv log_speech-recognition-community-v2_dev_data_es_validation_predictions.txt log_speech-recognition-community-v2_dev_data_es_validation_predictions_greedy.txt
mv speech-recognition-community-v2_dev_data_es_validation_eval_results.txt speech-recognition-community-v2_dev_data_es_validation_eval_results_greedy.txt

python eval.py --model_id jonatasgrosman/wav2vec2-large-xlsr-53-spanish --dataset speech-recognition-community-v2/dev_data --config es --split validation --chunk_length_s 5.0 --stride_length_s 1.0 --log_outputs
