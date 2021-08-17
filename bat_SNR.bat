REM python "C:\Matlab Files\timer\timer_start_next_2.py"
REM python "C:\Matlab Files\timer\timer_start_from_file.py"

REM python gen_SNR_videos_NAOMi7_time.py 120s_30Hz_N=200_100mW_noise10+23_NA0.8,0.6_GCaMP6f
REM python gen_SNR_videos_NAOMi7_time.py 120s_30Hz_N=200_100mW_noise10+23_NA0.8,0.6_GCaMP6s
REM python gen_SNR_videos_NAOMi7_time.py 1100s_3Hz_N=200_100mW_noise10+23_NA0.8,0.6_jGCaMP7f
REM python gen_SNR_videos_NAOMi7_time.py 1100s_3Hz_N=200_100mW_noise10+23_NA0.8,0.6_jGCaMP7s
REM python gen_SNR_videos_NAOMi7_time.py 1100s_3Hz_N=200_100mW_noise10+23_NA0.8,0.6_jGCaMP7b
REM python gen_SNR_videos_NAOMi7_time.py 1100s_3Hz_N=200_100mW_noise10+23_NA0.8,0.6_jGCaMP7c

REM python gen_SNR_videos_NAOMi_time.py 1000s_3Hz_N=200_40mW_noise10+23_NA0.4,0.3
REM python gen_SNR_videos_NAOMi7_time.py 110s_30Hz_N=200_100mW_noise10+23_NA0.8,0.6_GCaMP6f
REM python gen_SNR_videos_NAOMi7_time.py 110s_30Hz_N=200_100mW_noise10+23_NA0.8,0.6_GCaMP6s
REM python gen_SNR_videos_NAOMi7_time.py 1100s_3Hz_N=200_40mW_noise10+23_NA0.8,0.6_jGCaMP7c
REM python gen_SNR_videos_NAOMi7_time.py 1100s_3Hz_N=200_40mW_noise10+23_NA0.4,0.3_jGCaMP7s

REM python gen_SNR_videos_NAOMi_time.py 300s_10Hz_N=100_40mW_noise10+23_NA0.4,0.3
REM python gen_SNR_videos_NAOMi_time.py 300s_10Hz_N=200_40mW_noise10+23_NA0.4,0.3
REM python gen_SNR_videos_NAOMi_time.py 300s_10Hz_N=400_40mW_noise10+23
REM python gen_SNR_videos_NAOMi_time.py 1000s_3Hz_N=400_40mW_noise10+23
REM python gen_SNR_videos_NAOMi_time.py 100s_10Hz_N=400_40mW_noise10+23
REM python gen_SNR_videos_NAOMi_time.py 100s_3Hz_N=400_40mW_noise10+23
REM python gen_SNR_videos_NAOMi_time.py 100s_30Hz_N=400_40mW_noise10+23
REM python gen_SNR_videos_NAOMi_time.py 100s_30Hz_N=500_40mW_noise10+23

REM python gen_SNR_videos_arg_time.py 100s_30Hz_100+10_rand_vary-rate

REM python gen_SNR_videos_arg_time.py 100s_30Hz_100+10_vary-rate_lowbg
REM python gen_SNR_videos_arg_time.py 100s_30Hz_100+10_vary-rate
REM python gen_SNR_videos_arg_time.py 100s_30Hz_100+10_burst3
REM python gen_SNR_videos_arg_time.py 100s_30Hz_100+10_10%%
REM python gen_SNR_videos_arg_time.py 100s_30Hz_100+10_20%%
REM python gen_SNR_videos.py


python gen_SNR_videos.py
python gen_SNR_videos_ABO_full.py
python "C:\Matlab Files\timer\timer_stop.py"
REM shutdown -s -t 20
