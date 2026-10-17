# تثبيت خادم العرض الوهمي
!apt-get update && apt-get install -y xvfb

# تثبيت المكتبات المعتمدة (النسخ الحديثة تدعم gymnasium)
!pip install stable-baselines3[extra] gymnasium[mujoco] pyvirtualdisplay moviepy


import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from pyvirtualdisplay import Display

# 1. تشغيل الشاشة الوهمية (ضرورية في كاجل)
virtual_display = Display(visible=0, size=(1400, 900))
virtual_display.start()

# 2. إنشاء 8 بيئات متوازية تعمل في نفس الوقت!
env_id = "Hopper-v4"
# نستخدم SubprocVecEnv لتوزيع البيئات على أنوية المعالج الحقيقية
vec_env = make_vec_env(env_id, n_envs=8, vec_env_cls=SubprocVecEnv)

print("🚀 بدء تدريب Hopper باستخدام PPO مع 8 بيئات متوازية...")

# 3. بناء نموذج PPO بإعدادات قياسية ممتازة للقفز
model = PPO(
    "MlpPolicy",
    vec_env,
    learning_rate=0.0003,
    n_steps=2048,        # كل روبوت سيقوم بـ 2048 خطوة قبل التحديث
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,      # هذا هو سر PPO (الحد المحافظ الذي يمنع الانهيار)
    ent_coef=0.0,        # لا نحتاج لعشوائية إضافية، الـ Hopper يسقط بسرعة!
    verbose=1
)

# 4. التدريب لمدة 500,000 خطوة (ستنتهي بسرعة بفضل التوازي)
model.learn(total_timesteps=500000)

# 5. حفظ النموذج وإغلاق البيئات
model.save("ppo_hopper")
vec_env.close()
print("✅ تم تدريب الـ Hopper وحفظ النموذج بنجاح.")


# كود تسجيل الفيديو 

import os
import imageio
from IPython.display import Video, display
from pyvirtualdisplay import Display
from stable_baselines3 import PPO
import gymnasium as gym

# 1. تشغيل الشاشة الوهمية لتجنب أخطاء العرض
virtual_display = Display(visible=0, size=(1400, 900))
virtual_display.start()

# 2. الاستدعاء الصريح وضبط سرعة الإطارات الفيزيائية الصحيحة (125 FPS)
from gymnasium.envs.mujoco.hopper_v4 import HopperEnv
if "render_fps" in HopperEnv.metadata:
    HopperEnv.metadata["render_fps"] = 125

env_test = gym.make("Hopper-v4", render_mode="rgb_array")

# 3. تحميل دماغ الـ Hopper
model = PPO.load("ppo_hopper")

frames = []
state, _ = env_test.reset()
done = False
steps = 0

print("🎥 بدء تسجيل الـ Hopper وهو يقفز...")

# 4. حلقة التصوير
while not done and steps < 1000:
    frames.append(env_test.render())
    
    # اتخاذ القرار الحتمي (بدون استكشاف)
    action, _states = model.predict(state, deterministic=True)
    state, reward, terminated, truncated, _ = env_test.step(action)
    
    done = terminated or truncated
    steps += 1

env_test.close()

# 5. حفظ الفيديو بسرعة 125 إطار في الثانية
video_path = './mujoco_hopper_ppo.mp4'
imageio.mimsave(video_path, frames, fps=125, macro_block_size=None)
print(f"✅ تم تسجيل القفز بنجاح! إجمالي الإطارات: {len(frames)}")

# عرض الفيديو داخل دفتر كاجل
display(Video(video_path, embed=True))