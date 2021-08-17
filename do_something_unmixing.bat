python "C:\Matlab Files\timer\timer_start_next.py"

REM python nmfunmix_simulation_multi_arg.py 120s_30Hz_N=200_100mW_noise10+23_NA0.8,0.6_GCaMP6f Raw 1 1 0 0
REM python nmfunmix_simulation_multi_arg.py 120s_30Hz_N=200_100mW_noise10+23_NA0.8,0.6_GCaMP6f SNR 1 1 0 0

REM python run_FISSA_multi_simulation.py 120s_30Hz_N=200_100mW_noise10+23_NA0.8,0.6_GCaMP6f Raw
REM python run_FISSA_multi_simulation.py 120s_30Hz_N=200_100mW_noise10+23_NA0.8,0.6_GCaMP6f SNR

REM python AllenSDK_NAOMi_bgsubs_after.py 120s_30Hz_N=200_100mW_noise10+23_NA0.8,0.6_GCaMP6f Raw
REM python AllenSDK_NAOMi_bgsubs_after.py 120s_30Hz_N=200_100mW_noise10+23_NA0.8,0.6_GCaMP6f SNR
REM python AllenSDK_1p_bgsubs_after.py Raw
REM python AllenSDK_1p_bgsubs_after.py SNR
REM python AllenSDK_ABO_bgsubs_after.py Raw
REM python AllenSDK_ABO_bgsubs_after.py SNR

REM python nmfunmix_ABO_multi_novideounmix.py mean SNR 1 1 0 0
REM python nmfunmix_ABO_multi_novideounmix.py mean Raw 1 1 0 0
REM python nmfunmix_1p_multi_novideounmix.py mean SNR 1 1 0 0
REM python nmfunmix_1p_multi_novideounmix.py mean Raw 1 1 0 0

REM python run_FISSA_multi.py Raw
REM python run_FISSA_multi.py SNR
REM python run_FISSA_1p.py Raw
REM python run_FISSA_1p.py SNR

REM python run_FISSA_multi_simulation.py 120s_30Hz_N=200_100mW_noise10+23_NA0.8,0.6_GCaMP6f Raw
REM python run_FISSA_multi_simulation.py 120s_30Hz_N=200_100mW_noise10+23_NA0.8,0.6_GCaMP6f SNR
REM python nmfunmix_simulation_multi_arg_novideounmix.py mean Raw 1 1 0 0 120s_30Hz_N=200_100mW_noise10+23_NA0.8,0.6_GCaMP6f
REM python nmfunmix_simulation_multi_arg_novideounmix.py mean SNR 1 1 0 0 120s_30Hz_N=200_100mW_noise10+23_NA0.8,0.6_GCaMP6f

REM python nmfunmix_ABO_novideounmix.py mean SNR 1 1 0 0
REM python nmfunmix_ABO.py mean Raw 1 1 0 0
REM python run_FISSA_ABO.py Raw
REM python run_FISSA_ABO.py SNR
REM python nmfunmix_ABO_full_multi.py mean SNR 1 1 0 0
REM python nmfunmix_ABO_full_multi.py mean Raw 1 1 0 0
python AllenSDK_ABO_full_bgsubs_after.py Raw
python AllenSDK_ABO_full_bgsubs_after.py SNR

python "C:\Matlab Files\timer\timer_stop.py"

REM shutdown -s -t 60
REM shutdown -a

REM python nmfunmix_ABO_multi_novideounmix.py mean Raw 2 1 0 0
REM python nmfunmix_1p_multi_novideounmix.py mean Raw 2 1 0 0
REM python nmfunmix_simulation_multi_arg_novideounmix.py mean Raw 2 1 0 0 120s_30Hz_N=200_100mW_noise10+23_NA0.8,0.6_GCaMP6f
