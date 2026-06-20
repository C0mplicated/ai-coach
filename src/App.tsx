import { useEffect, useMemo, useRef, useState } from 'react';
import { motion } from 'framer-motion';
import { Area, AreaChart, ResponsiveContainer, XAxis, Tooltip } from 'recharts';
import * as poseDetection from '@tensorflow-models/pose-detection';
import '@tensorflow/tfjs-backend-webgl';
import * as tf from '@tensorflow/tfjs-core';
import { useAnalysisStore, type AnalysisState } from './store/useAnalysisStore';
import {
  getCoachingFeedback,
  getJointAngle,
  getPoseConfidence,
  metricDefinitions,
  MetricKey,
  sampleVideoUrl,
  supportedVideoTypes,
} from './lib/poseUtils';

const playerDelay = 120;

const formatTime = (seconds: number) => {
  if (!Number.isFinite(seconds)) return '0:00';
  const minutes = Math.floor(seconds / 60);
  const remainder = Math.floor(seconds % 60);
  return `${minutes}:${remainder.toString().padStart(2, '0')}`;
};

const getVideoSourceLabel = (url: string) => {
  if (!url) return 'No video loaded';
  if (url.startsWith('blob:')) return 'Local upload';
  return 'Remote URL';
};

const drawSkeleton = (ctx: CanvasRenderingContext2D, keypoints: poseDetection.Keypoint[]) => {
  const pairs = poseDetection.util.getAdjacentPairs(poseDetection.SupportedModels.MoveNet);
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);

  ctx.strokeStyle = 'rgba(56, 189, 248, 0.86)';
  ctx.lineWidth = 3;
  ctx.lineJoin = 'round';

  pairs.forEach(([i, j]) => {
    const a = keypoints[i];
    const b = keypoints[j];
    if (a.score && b.score && a.score > 0.35 && b.score > 0.35) {
      ctx.beginPath();
      ctx.moveTo(a.x ?? 0, a.y ?? 0);
      ctx.lineTo(b.x ?? 0, b.y ?? 0);
      ctx.stroke();
    }
  });

  keypoints.forEach((point) => {
    if (point.score && point.score > 0.35) {
      ctx.fillStyle = 'rgba(56, 189, 248, 0.95)';
      ctx.beginPath();
      ctx.arc(point.x ?? 0, point.y ?? 0, 4, 0, Math.PI * 2);
      ctx.fill();
    }
  });
};

const buildChartData = (history: number[]) => history.map((value, index) => ({ index, value }));

export default function App() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const detectorRef = useRef<poseDetection.PoseDetector | null>(null);
  const intervalRef = useRef<number | null>(null);

  const videoUrl = useAnalysisStore((state: AnalysisState) => state.videoUrl);
  const metrics = useAnalysisStore((state: AnalysisState) => state.metrics);
  const poseConfidence = useAnalysisStore((state: AnalysisState) => state.poseConfidence);
  const status = useAnalysisStore((state: AnalysisState) => state.status);
  const isAnalyzing = useAnalysisStore((state: AnalysisState) => state.isAnalyzing);

  const setVideoUrl = useAnalysisStore((state: AnalysisState) => state.setVideoUrl);
  const pushAngle = useAnalysisStore((state: AnalysisState) => state.pushAngle);
  const resetAnalysis = useAnalysisStore((state: AnalysisState) => state.resetAnalysis);
  const setPoseConfidence = useAnalysisStore((state: AnalysisState) => state.setPoseConfidence);
  const setFeedback = useAnalysisStore((state: AnalysisState) => state.setFeedback);
  const setStatus = useAnalysisStore((state: AnalysisState) => state.setStatus);
  const setAnalyzing = useAnalysisStore((state: AnalysisState) => state.setAnalyzing);

  const [sourceUrl, setSourceUrl] = useState('');
  const [duration, setDuration] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  const [dragActive, setDragActive] = useState(false);

  const currentFeedback = useMemo(
    () => getCoachingFeedback({ metrics, poseConfidence }),
    [metrics, poseConfidence]
  );

  const analysisStats = useMemo(
    () =>
      (Object.keys(metrics) as MetricKey[]).map((key) => ({
        key,
        label: metricDefinitions[key].label,
        color: metricDefinitions[key].color,
        current: metrics[key].current,
        avg: metrics[key].avg,
        min: metrics[key].min,
        max: metrics[key].max,
        history: buildChartData(metrics[key].history),
      })),
    [metrics]
  );

  const finalReport = useMemo(() => {
    if (!videoUrl) {
      return {
        title: 'No video analyzed yet',
        bullets: ['Upload a file or paste a shareable URL to generate a motion summary.'],
      };
    }

    const bullets: string[] = [];
    if (poseConfidence < 0.35) {
      bullets.push('Pose confidence is low. Use brighter lighting and a clearer camera angle.');
    }

    const kneeCurrent = metrics.knee.current;
    const hipCurrent = metrics.hip.current;
    const shoulderCurrent = metrics.shoulder.current;
    const elbowCurrent = metrics.elbow.current;

    if (kneeCurrent !== null) {
      if (kneeCurrent < 70) {
        bullets.push('Knee bend is too shallow. Increase flexion for a stronger lunge.');
      } else if (kneeCurrent > 115) {
        bullets.push('Knee bend looks good. Keep a stable center of gravity during recovery.');
      } else {
        bullets.push('Knee angle is within a solid range for effective footwork.');
      }
    }

    if (hipCurrent !== null) {
      if (hipCurrent < 60) {
        bullets.push('Hip angle is tight. Open your hips slightly to improve balance.');
      } else {
        bullets.push('Hip movement is stable, supporting a controlled shot.');
      }
    }

    if (shoulderCurrent !== null && shoulderCurrent < 45) {
      bullets.push('Shoulder rotation is limited. Try keeping your upper body more open.');
    }

    if (elbowCurrent !== null && elbowCurrent > 170) {
      bullets.push('Elbow extension is very straight. Add a bit of bend for a smoother follow-through.');
    }

    if (bullets.length === 0) {
      bullets.push('Your motion looks consistent. Continue refining flow and stability across the movement.');
    }

    return {
      title: 'Final motion report',
      bullets,
    };
  }, [videoUrl, metrics, poseConfidence]);

  useEffect(() => {
    let active = true;
    const loadModel = async () => {
      try {
        await tf.setBackend('webgl');
        await tf.ready();
        detectorRef.current = await poseDetection.createDetector(
          poseDetection.SupportedModels.MoveNet,
          { modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING }
        );
        if (active) {
          setStatus('Ready to analyze badminton motion.');
        }
      } catch (error) {
        console.error(error);
        setStatus('Model load failed. Check console for details.');
      }
    };
    loadModel();
    return () => {
      active = false;
      detectorRef.current?.dispose();
      detectorRef.current = null;
    };
  }, [setStatus]);

  useEffect(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || !detectorRef.current) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const analyzeFrame = async () => {
      if (!video || video.paused || video.ended) return;
      const poses = await detectorRef.current!.estimatePoses(video, { maxPoses: 1, flipHorizontal: false });
      if (poses.length === 0) {
        setFeedback('No pose detected. Please choose a clearer side view.');
        return;
      }

      const pose = poses[0];
      drawSkeleton(ctx, pose.keypoints);
      const confidence = getPoseConfidence(pose.keypoints);
      setPoseConfidence(confidence);

      (Object.keys(metricDefinitions) as MetricKey[]).forEach((key) => {
        const angle = getJointAngle(key, pose.keypoints);
        if (angle !== null) {
          pushAngle(key, angle);
        }
      });

      const knee = getJointAngle('knee', pose.keypoints);
      setFeedback(
        knee === null
          ? 'Pose detected, but knee joints are not fully visible yet.'
          : `Current knee angle ${Math.round(knee)}°. Keep tracking form through each frame.`
      );
    };

    if (intervalRef.current !== null) window.clearInterval(intervalRef.current);
    intervalRef.current = window.setInterval(() => {
      analyzeFrame().catch((error) => console.error(error));
    }, playerDelay);

    return () => {
      if (intervalRef.current !== null) window.clearInterval(intervalRef.current);
    };
  }, [videoUrl, pushAngle, setFeedback, setPoseConfidence]);

  const loadVideoFromFile = (file: File) => {
    const url = URL.createObjectURL(file);
    setVideoUrl(url);
    setSourceUrl('');
    resetAnalysis();
    setAnalyzing(true);
    setStatus('Video loaded. Analyzing pose data...');
  };

  const handleUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    loadVideoFromFile(file);
  };

  const handleDrop = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setDragActive(false);
    const file = event.dataTransfer.files?.[0];
    if (file) loadVideoFromFile(file);
  };

  const handleLoadUrl = () => {
    const trimmed = sourceUrl.trim();
    if (!trimmed) return;
    setVideoUrl(trimmed);
    resetAnalysis();
    setAnalyzing(true);
    setStatus('Remote video loaded. Starting pose analysis...');
  };

  const handleExample = () => {
    setVideoUrl(sampleVideoUrl);
    resetAnalysis();
    setAnalyzing(true);
    setStatus('Loaded demo video. Tracking pose now.');
  };

  const handleTimeUpdate = () => {
    const video = videoRef.current;
    if (!video) return;
    setCurrentTime(video.currentTime);
  };

  const handleDuration = () => {
    const video = videoRef.current;
    if (!video) return;
    setDuration(video.duration);
    const canvas = canvasRef.current;
    if (canvas) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
    }
  };

  const seekVideo = (value: number) => {
    const video = videoRef.current;
    if (!video) return;
    video.currentTime = value;
    setCurrentTime(value);
  };

  const stepFrame = (direction: 'back' | 'forward') => {
    const video = videoRef.current;
    if (!video) return;
    const step = 1 / 30;
    const nextTime = Math.max(0, Math.min(video.duration, video.currentTime + (direction === 'forward' ? step : -step)));
    video.currentTime = nextTime;
    setCurrentTime(nextTime);
  };

  const percent = duration ? (currentTime / duration) * 100 : 0;

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <div className="mx-auto max-w-7xl px-6 py-8 sm:px-8 lg:px-10">
        <motion.header
          initial={{ opacity: 0, y: 24 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.55 }}
          className="rounded-[34px] border border-white/10 bg-slate-950/95 p-8 shadow-[0_40px_120px_rgba(15,23,42,0.55)] backdrop-blur-3xl"
        >
          <div className="flex flex-col gap-8 xl:flex-row xl:items-center xl:justify-between">
            <div className="max-w-2xl space-y-6">
              <div className="inline-flex items-center gap-3 rounded-full bg-cyan-500/10 px-4 py-2 text-sm font-semibold uppercase tracking-[0.24em] text-cyan-200 shadow-[0_12px_40px_rgba(56,189,248,0.08)]">
                <span className="flex h-9 w-9 items-center justify-center rounded-full bg-cyan-500/15 text-cyan-200">MC</span>
                Motion Coach AI
              </div>
              <div className="space-y-4">
                <h1 className="text-4xl font-semibold tracking-tight text-slate-50 sm:text-5xl">
                  Next-level badminton motion analysis.
                </h1>
                <p className="max-w-2xl text-base leading-8 text-slate-300 sm:text-lg">
                  Upload a video or paste a shared URL to analyze movement, detect pose, overlay a skeleton, and get beginner-friendly coaching insights.
                </p>
              </div>
            </div>

            <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-1">
              <div className="rounded-[28px] border border-white/10 bg-slate-950/80 p-5 shadow-[0_24px_64px_rgba(15,23,42,0.18)]">
                <p className="text-sm uppercase tracking-[0.24em] text-slate-400">Current status</p>
                <p className="mt-4 text-lg font-semibold text-slate-100">{status}</p>
              </div>
              <div className="rounded-[28px] border border-white/10 bg-slate-950/80 p-5 shadow-[0_24px_64px_rgba(15,23,42,0.18)]">
                <p className="text-sm uppercase tracking-[0.24em] text-slate-400">Confidence</p>
                <p className="mt-4 text-lg font-semibold text-cyan-300">{(poseConfidence * 100).toFixed(0)}%</p>
              </div>
            </div>
          </div>
        </motion.header>

        <div className="mt-8 grid gap-8 xl:grid-cols-[1.3fr_0.95fr]">
          <motion.section
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.55, ease: 'easeOut' }}
            className="space-y-6 rounded-[32px] border border-white/10 bg-slate-950/90 p-6 shadow-[0_40px_80px_rgba(15,23,42,0.35)] backdrop-blur-3xl"
          >
            <div className="grid gap-4 sm:grid-cols-2">
              <div className="rounded-[28px] border border-white/10 bg-slate-950/90 p-6 shadow-[0_24px_48px_rgba(15,23,42,0.18)] transition duration-300 ease-out hover:-translate-y-1 hover:shadow-[0_30px_60px_rgba(15,23,42,0.18)]">
                <p className="text-sm uppercase tracking-[0.24em] text-slate-400">Video input</p>
                <div className="mt-5 space-y-4">
                  <label className="block cursor-pointer rounded-[28px] border border-slate-700/80 bg-slate-950/95 px-5 py-6 text-center transition duration-300 ease-out hover:border-cyan-400/70 hover:bg-slate-900/95">
                    <span className="block text-sm font-semibold text-slate-100">Upload file</span>
                    <span className="mt-3 block text-xs text-slate-400">MP4, MOV, WebM</span>
                    <input className="sr-only" type="file" accept={supportedVideoTypes.join(',')} onChange={handleUpload} />
                  </label>
                  <div className="rounded-[28px] border border-slate-700/80 bg-slate-950/95 p-4 transition duration-300 ease-out hover:border-cyan-400/60">
                    <label className="text-sm font-semibold text-slate-200">Video URL</label>
                    <div className="mt-3 flex gap-3">
                      <input
                        value={sourceUrl}
                        onChange={(event) => setSourceUrl(event.target.value)}
                        placeholder="https://..."
                        className="w-full rounded-2xl border border-slate-700 bg-slate-900 px-4 py-3 text-sm text-slate-100 outline-none transition duration-300 ease-out focus:border-cyan-400/80 focus:ring-2 focus:ring-cyan-400/15"
                      />
                      <motion.button
                        whileHover={{ y: -2 }}
                        whileTap={{ scale: 0.98 }}
                        onClick={handleLoadUrl}
                        className="rounded-2xl bg-cyan-500 px-4 py-3 text-sm font-semibold text-slate-950 transition duration-300 hover:bg-cyan-400"
                      >
                        Load
                      </motion.button>
                    </div>
                  </div>
                </div>
              </div>

              <motion.div
                whileHover={{ y: -2 }}
                className="rounded-[28px] border border-white/10 bg-slate-950/90 p-6 shadow-[0_24px_48px_rgba(15,23,42,0.18)] transition duration-300 ease-out"
              >
                <p className="text-sm uppercase tracking-[0.24em] text-slate-400">Quick demo</p>
                <p className="mt-4 text-sm leading-6 text-slate-300">
                  Launch a sample sequence instantly or drag a file directly into the player.
                </p>
                <motion.button
                  whileHover={{ y: -2 }}
                  whileTap={{ scale: 0.98 }}
                  onClick={handleExample}
                  className="mt-6 inline-flex items-center justify-center rounded-2xl bg-cyan-500 px-5 py-3 text-sm font-semibold text-slate-950 transition duration-300 hover:bg-cyan-400"
                >
                  Load Demo Video
                </motion.button>
              </motion.div>
            </div>

            <motion.div
              onDrop={handleDrop}
              onDragOver={(event) => {
                event.preventDefault();
                setDragActive(true);
              }}
              onDragLeave={() => setDragActive(false)}
              whileHover={{ y: -2 }}
              className={`group relative overflow-hidden rounded-[28px] border-2 ${
                dragActive ? 'border-cyan-400/90 bg-cyan-400/10 shadow-[0_0_0_1px_rgba(56,189,248,0.35)]' : 'border-dashed border-slate-700/70 bg-slate-950/85'
              } transition-all duration-300 ease-out`}
            >
              <div className="absolute inset-0 bg-gradient-to-br from-cyan-500/5 via-transparent to-slate-950/0 opacity-80" />
              <div className="relative flex min-h-[240px] flex-col items-center justify-center gap-4 p-8 text-center transition duration-300 ease-out">
                <p className="text-sm uppercase tracking-[0.2em] text-cyan-300/90">Drag & drop</p>
                <h2 className="text-2xl font-semibold text-slate-50">Drop a badminton clip here</h2>
                <p className="max-w-xl text-sm leading-6 text-slate-400">
                  Supports MP4, MOV, and WebM files. Or paste a publicly accessible video URL above.
                </p>
              </div>
            </motion.div>
          </motion.section>

          <aside className="space-y-6 rounded-[32px] border border-slate-700/80 bg-slate-900/75 p-6 shadow-[0_40px_80px_rgba(15,23,42,0.4)] backdrop-blur-xl">
            <div className="rounded-3xl border border-slate-700/80 bg-slate-950/80 p-5">
              <p className="text-sm uppercase tracking-[0.24em] text-slate-500">Live coaching</p>
              <p className="mt-4 text-lg font-semibold text-slate-100">{currentFeedback}</p>
            </div>

            <div className="grid gap-4 sm:grid-cols-2">
              <div className="rounded-3xl border border-slate-700/80 bg-slate-950/80 p-5">
                <p className="text-sm uppercase tracking-[0.24em] text-slate-500">Video source</p>
                <p className="mt-3 text-base text-slate-100">{getVideoSourceLabel(videoUrl)}</p>
              </div>
              <div className="rounded-3xl border border-slate-700/80 bg-slate-950/80 p-5">
                <p className="text-sm uppercase tracking-[0.24em] text-slate-500">Playback</p>
                <p className="mt-3 text-base text-slate-100">{formatTime(currentTime)} / {formatTime(duration)}</p>
              </div>
            </div>

            <div className="rounded-3xl border border-slate-700/80 bg-slate-950/80 p-5">
              <p className="text-sm uppercase tracking-[0.24em] text-slate-500">Controls</p>
              <div className="mt-4 flex gap-3">
                <button
                  onClick={() => stepFrame('back')}
                  className="flex-1 rounded-2xl border border-slate-700/90 bg-slate-900 px-4 py-3 text-sm font-semibold text-slate-100 transition hover:border-cyan-400/70"
                >
                  Previous frame
                </button>
                <button
                  onClick={() => stepFrame('forward')}
                  className="flex-1 rounded-2xl border border-slate-700/90 bg-slate-900 px-4 py-3 text-sm font-semibold text-slate-100 transition hover:border-cyan-400/70"
                >
                  Next frame
                </button>
              </div>
            </div>
          </aside>
        </div>

        <motion.section
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.55, ease: 'easeOut' }}
          className="mt-8 grid gap-6 md:grid-cols-2 xl:grid-cols-4"
        >
          {analysisStats.map((metric) => (
            <motion.div
              key={metric.key}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              whileHover={{ y: -4 }}
              whileTap={{ scale: 0.99 }}
              transition={{ duration: 0.3, ease: 'easeOut' }}
              className="rounded-[28px] border border-white/10 bg-slate-950/95 p-5 shadow-[0_20px_40px_rgba(15,23,42,0.18)]"
            >
              <div className="flex items-start justify-between gap-3">
                <div>
                  <p className="text-sm uppercase tracking-[0.24em] text-slate-400">{metric.label}</p>
                  <p className="mt-3 text-3xl font-semibold text-slate-100">
                    {metric.current !== null ? `${Math.round(metric.current)}°` : '—'}
                  </p>
                </div>
                <div className="rounded-2xl bg-slate-900/95 px-3 py-2 text-sm font-semibold text-slate-100" style={{ color: metric.color }}>
                  {metric.current !== null ? (metric.current > 120 ? 'High' : metric.current < 70 ? 'Low' : 'Ideal') : 'Waiting'}
                </div>
              </div>
              <div className="mt-5 grid gap-2 text-xs uppercase tracking-[0.24em] text-slate-500 sm:grid-cols-3">
                <span>Avg {Math.round(metric.avg)}°</span>
                <span>Min {Math.round(metric.min)}°</span>
                <span>Max {Math.round(metric.max)}°</span>
              </div>
              <div className="mt-4 h-24">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={metric.history}>
                    <XAxis dataKey="index" hide />
                    <Tooltip contentStyle={{ background: '#0f172a', border: 'none' }} formatter={(value: number) => [`${Math.round(value)}°`, metric.label]} />
                    <Area type="monotone" dataKey="value" stroke={metric.color} fill={metric.color} fillOpacity={0.2} strokeWidth={2} />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </motion.div>
          ))}
        </motion.section>

        <section className="mt-8 overflow-hidden rounded-[32px] border border-white/10 bg-slate-950/90 p-6 shadow-[0_40px_80px_rgba(15,23,42,0.35)] backdrop-blur-3xl">
          <div className="grid gap-6 xl:grid-cols-[1.5fr_0.75fr]">
            <div className="rounded-[28px] border border-white/10 bg-slate-950/95 p-4 shadow-[0_24px_48px_rgba(15,23,42,0.18)] transition duration-300 ease-out hover:-translate-y-1 hover:shadow-[0_30px_60px_rgba(15,23,42,0.22)]">
              <div className="relative overflow-hidden rounded-[24px] bg-slate-950 transition duration-300 ease-out">
                <div className="absolute left-4 top-4 z-10 rounded-full bg-slate-950/90 px-3 py-1 text-xs uppercase tracking-[0.24em] text-cyan-300/90 shadow-[0_8px_24px_rgba(0,0,0,0.25)]">
                  Pose overlay live
                </div>
                <video
                  ref={videoRef}
                  src={videoUrl}
                  controls
                  muted
                  playsInline
                  crossOrigin="anonymous"
                  onTimeUpdate={handleTimeUpdate}
                  onLoadedMetadata={handleDuration}
                  className="h-full w-full bg-slate-950 transition duration-300 ease-out"
                />
                <canvas ref={canvasRef} className="pointer-events-none absolute inset-0 h-full w-full" />
              </div>
              <div className="mt-4 rounded-[24px] border border-white/10 bg-slate-900/80 p-4">
                <div className="mb-3 flex items-center justify-between gap-4 text-sm text-slate-400">
                  <span>Playback scrubber</span>
                  <span>{Math.round(percent)}%</span>
                </div>
                <input
                  type="range"
                  min={0}
                  max={duration || 0}
                  step={0.01}
                  value={currentTime}
                  onChange={(event) => seekVideo(Number(event.target.value))}
                  className="h-2 w-full cursor-pointer appearance-none rounded-full bg-slate-700 accent-cyan-400"
                />
              </div>
            </div>

            <div className="rounded-[28px] border border-white/10 bg-slate-950/95 p-6 shadow-[0_24px_48px_rgba(15,23,42,0.18)]">
              <p className="text-sm uppercase tracking-[0.24em] text-cyan-300/90">Analysis snapshot</p>
              <div className="mt-6 space-y-4 text-slate-300">
                <div className="grid gap-4 sm:grid-cols-2">
                  <div className="rounded-[24px] bg-slate-900/85 p-4">
                    <p className="text-xs uppercase tracking-[0.24em] text-slate-500">Pose status</p>
                    <p className="mt-2 text-lg font-semibold text-slate-100">{isAnalyzing ? 'Analyzing' : 'Idle'}</p>
                  </div>
                  <div className="rounded-[24px] bg-slate-900/85 p-4">
                    <p className="text-xs uppercase tracking-[0.24em] text-slate-500">Analysis frames</p>
                    <p className="mt-2 text-lg font-semibold text-slate-100">{metrics.knee.history.length}</p>
                  </div>
                </div>
                <div className="rounded-[24px] bg-slate-900/85 p-4">
                  <p className="text-xs uppercase tracking-[0.24em] text-slate-500">Player notes</p>
                  <p className="mt-2 text-sm leading-7 text-slate-300">
                    {videoUrl ? 'Use the scrubber and frame controls to inspect technique. Metrics update while the video plays.' : 'Load a video to unlock joint analytics and coaching recommendations.'}
                  </p>
                </div>
              </div>
            </div>
          </div>
        </section>

        <motion.section
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.55, ease: 'easeOut' }}
          className="mt-8 rounded-[32px] border border-white/10 bg-slate-950/90 p-6 shadow-[0_40px_80px_rgba(15,23,42,0.35)] backdrop-blur-3xl"
        >
          <div className="space-y-5">
            <div className="flex items-center justify-between gap-4">
              <div>
                <p className="text-sm uppercase tracking-[0.24em] text-cyan-300/90">Final report</p>
                <h2 className="mt-2 text-2xl font-semibold text-slate-100">Summary & improvement plan</h2>
              </div>
              <span className="rounded-full bg-slate-900/80 px-4 py-2 text-sm font-semibold text-slate-200">Confidence {Math.round(poseConfidence * 100)}%</span>
            </div>

            <div className="grid gap-4 lg:grid-cols-[1fr_0.85fr]">
              <div className="rounded-[28px] border border-white/10 bg-slate-900/85 p-5 shadow-[0_20px_40px_rgba(15,23,42,0.18)]">
                <p className="text-sm uppercase tracking-[0.24em] text-slate-400">Key takeaways</p>
                <ul className="mt-4 space-y-3 text-sm text-slate-200">
                  {finalReport.bullets.map((item, index) => (
                    <li key={index} className="flex gap-3">
                      <span className="mt-1 inline-flex h-2.5 w-2.5 rounded-full bg-cyan-400" />
                      <span>{item}</span>
                    </li>
                  ))}
                </ul>
              </div>
              <div className="rounded-[28px] border border-white/10 bg-slate-900/85 p-5 shadow-[0_20px_40px_rgba(15,23,42,0.18)]">
                <p className="text-sm uppercase tracking-[0.24em] text-slate-400">Report details</p>
                <div className="mt-5 space-y-3 text-sm text-slate-300">
                  <div className="rounded-3xl bg-slate-950/85 p-4">
                    <p className="text-xs uppercase tracking-[0.24em] text-slate-500">Knee current</p>
                    <p className="mt-2 text-lg font-semibold text-slate-100">{metrics.knee.current !== null ? `${Math.round(metrics.knee.current)}°` : '—'}</p>
                  </div>
                  <div className="rounded-3xl bg-slate-950/85 p-4">
                    <p className="text-xs uppercase tracking-[0.24em] text-slate-500">Hip current</p>
                    <p className="mt-2 text-lg font-semibold text-slate-100">{metrics.hip.current !== null ? `${Math.round(metrics.hip.current)}°` : '—'}</p>
                  </div>
                  <div className="rounded-3xl bg-slate-950/85 p-4">
                    <p className="text-xs uppercase tracking-[0.24em] text-slate-500">Shoulder current</p>
                    <p className="mt-2 text-lg font-semibold text-slate-100">{metrics.shoulder.current !== null ? `${Math.round(metrics.shoulder.current)}°` : '—'}</p>
                  </div>
                  <div className="rounded-3xl bg-slate-950/85 p-4">
                    <p className="text-xs uppercase tracking-[0.24em] text-slate-500">Elbow current</p>
                    <p className="mt-2 text-lg font-semibold text-slate-100">{metrics.elbow.current !== null ? `${Math.round(metrics.elbow.current)}°` : '—'}</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </motion.section>
      </div>
    </div>
  );
}
