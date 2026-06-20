import { useEffect, useMemo, useRef, useState } from 'react';
import { motion } from 'framer-motion';
import { Area, AreaChart, ResponsiveContainer, Tooltip, XAxis } from 'recharts';
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
import heroImage from './assets/motion-coach-hero.png';

type Keypoint = import('@tensorflow-models/pose-detection').Keypoint;
type PoseDetector = import('@tensorflow-models/pose-detection').PoseDetector;
type PoseDetectionModule = typeof import('@tensorflow-models/pose-detection');

const playerDelay = 120;

const formatTime = (seconds: number) => {
  if (!Number.isFinite(seconds)) return '0:00';
  const minutes = Math.floor(seconds / 60);
  const remainder = Math.floor(seconds % 60);
  return `${minutes}:${remainder.toString().padStart(2, '0')}`;
};

const formatAngle = (value: number | null) => (value === null ? '-' : `${Math.round(value)} deg`);

const getVideoSourceLabel = (url: string) => {
  if (!url) return 'No video loaded';
  if (url.startsWith('blob:')) return 'Local upload';
  return 'Remote URL';
};

const buildChartData = (history: number[]) => history.map((value, index) => ({ index, value }));

const drawSkeleton = (
  ctx: CanvasRenderingContext2D,
  keypoints: Keypoint[],
  pairs: Array<[number, number]>
) => {
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  ctx.strokeStyle = 'rgba(0, 113, 227, 0.88)';
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
      ctx.fillStyle = 'rgba(0, 113, 227, 0.96)';
      ctx.beginPath();
      ctx.arc(point.x ?? 0, point.y ?? 0, 4.5, 0, Math.PI * 2);
      ctx.fill();
    }
  });
};

export default function App() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const detectorRef = useRef<PoseDetector | null>(null);
  const poseModuleRef = useRef<PoseDetectionModule | null>(null);
  const adjacentPairsRef = useRef<Array<[number, number]>>([]);
  const intervalRef = useRef<number | null>(null);
  const objectUrlRef = useRef<string | null>(null);

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
  const [modelLoading, setModelLoading] = useState(false);

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
        title: 'No motion session yet',
        bullets: ['Load a badminton clip to generate a coaching summary.'],
      };
    }

    const bullets: string[] = [];
    if (poseConfidence < 0.35) {
      bullets.push('Pose confidence is low. Use brighter lighting and keep the full body in frame.');
    }

    const kneeCurrent = metrics.knee.current;
    const hipCurrent = metrics.hip.current;
    const shoulderCurrent = metrics.shoulder.current;
    const elbowCurrent = metrics.elbow.current;

    if (kneeCurrent !== null) {
      if (kneeCurrent < 70) {
        bullets.push('Knee bend is shallow. Increase flexion for a stronger lunge base.');
      } else if (kneeCurrent > 115) {
        bullets.push('Knee bend looks strong. Keep your center of gravity stable during recovery.');
      } else {
        bullets.push('Knee angle is in a solid range for controlled badminton footwork.');
      }
    }

    if (hipCurrent !== null) {
      bullets.push(
        hipCurrent < 60
          ? 'Hip angle is tight. Open the hips slightly to improve balance.'
          : 'Hip movement looks stable and supports controlled shot preparation.'
      );
    }

    if (shoulderCurrent !== null && shoulderCurrent < 45) {
      bullets.push('Shoulder rotation is limited. Keep the upper body more open through the shot.');
    }

    if (elbowCurrent !== null && elbowCurrent > 170) {
      bullets.push('Elbow extension is very straight. Add a small bend for a smoother follow-through.');
    }

    if (bullets.length === 0) {
      bullets.push('Your movement looks consistent. Keep refining rhythm, flow, and recovery timing.');
    }

    return {
      title: 'Motion report',
      bullets,
    };
  }, [videoUrl, metrics, poseConfidence]);

  useEffect(() => {
    if (!videoUrl || detectorRef.current || modelLoading) return;

    let active = true;
    const loadModel = async () => {
      try {
        setModelLoading(true);
        setStatus('Preparing the pose model...');
        const [tf, poseDetection] = await Promise.all([
          import('@tensorflow/tfjs-core'),
          import('@tensorflow-models/pose-detection'),
          import('@tensorflow/tfjs-backend-webgl'),
        ]).then(([tfModule, poseModule]) => [tfModule, poseModule] as const);

        await tf.setBackend('webgl');
        await tf.ready();

        const detector = await poseDetection.createDetector(
          poseDetection.SupportedModels.MoveNet,
          { modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING }
        );

        if (!active) {
          detector.dispose();
          return;
        }

        poseModuleRef.current = poseDetection;
        detectorRef.current = detector;
        adjacentPairsRef.current = poseDetection.util.getAdjacentPairs(
          poseDetection.SupportedModels.MoveNet
        ) as Array<[number, number]>;
        setStatus('Ready. Press play to analyze motion.');
      } catch (error) {
        console.error(error);
        setStatus('Model load failed. Check the console for details.');
      } finally {
        if (active) setModelLoading(false);
      }
    };

    loadModel();

    return () => {
      active = false;
    };
  }, [modelLoading, setStatus, videoUrl]);

  useEffect(() => {
    return () => {
      if (intervalRef.current !== null) window.clearInterval(intervalRef.current);
      detectorRef.current?.dispose();
      if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
    };
  }, []);

  useEffect(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || !detectorRef.current || !poseModuleRef.current) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const analyzeFrame = async () => {
      if (!video || video.paused || video.ended || !detectorRef.current) return;

      const poses = await detectorRef.current.estimatePoses(video, {
        maxPoses: 1,
        flipHorizontal: false,
      });

      if (poses.length === 0) {
        setFeedback('No pose detected. Try a clearer side view with the full body visible.');
        return;
      }

      const pose = poses[0];
      drawSkeleton(ctx, pose.keypoints, adjacentPairsRef.current);
      const confidence = getPoseConfidence(pose.keypoints);
      setPoseConfidence(confidence);

      (Object.keys(metricDefinitions) as MetricKey[]).forEach((key) => {
        const angle = getJointAngle(key, pose.keypoints);
        if (angle !== null) pushAngle(key, angle);
      });

      const knee = getJointAngle('knee', pose.keypoints);
      setFeedback(
        knee === null
          ? 'Pose detected, but the knee joints are not fully visible yet.'
          : `Current knee angle ${Math.round(knee)} deg. Keep tracking form through each frame.`
      );
    };

    if (intervalRef.current !== null) window.clearInterval(intervalRef.current);
    intervalRef.current = window.setInterval(() => {
      analyzeFrame().catch((error) => console.error(error));
    }, playerDelay);

    return () => {
      if (intervalRef.current !== null) window.clearInterval(intervalRef.current);
    };
  }, [videoUrl, pushAngle, setFeedback, setPoseConfidence, modelLoading]);

  const prepareNewVideo = (url: string, statusText: string) => {
    setVideoUrl(url);
    resetAnalysis();
    setAnalyzing(true);
    setCurrentTime(0);
    setDuration(0);
    setStatus(statusText);
    window.setTimeout(() => {
      document.getElementById('analyzer')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 80);
  };

  const loadVideoFromFile = (file: File) => {
    if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
    const url = URL.createObjectURL(file);
    objectUrlRef.current = url;
    setSourceUrl('');
    prepareNewVideo(url, 'Video loaded. Preparing AI motion analysis...');
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
    prepareNewVideo(trimmed, 'Remote video loaded. Preparing AI motion analysis...');
  };

  const handleExample = () => {
    prepareNewVideo(sampleVideoUrl, 'Demo video loaded. Preparing AI motion analysis...');
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

  const downloadReport = () => {
    const lines = [
      'Motion Coach AI Report',
      `Confidence: ${Math.round(poseConfidence * 100)}%`,
      `Source: ${getVideoSourceLabel(videoUrl)}`,
      '',
      ...finalReport.bullets.map((item) => `- ${item}`),
      '',
      'Current angles:',
      `Knee: ${formatAngle(metrics.knee.current)}`,
      `Hip: ${formatAngle(metrics.hip.current)}`,
      `Shoulder: ${formatAngle(metrics.shoulder.current)}`,
      `Elbow: ${formatAngle(metrics.elbow.current)}`,
    ];
    const blob = new Blob([lines.join('\n')], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'motion-coach-report.txt';
    link.click();
    URL.revokeObjectURL(url);
  };

  const percent = duration ? (currentTime / duration) * 100 : 0;

  return (
    <div className="min-h-screen bg-[#f5f5f7] text-[#1d1d1f]">
      <header className="sticky top-0 z-50 border-b border-black/5 bg-white/75 backdrop-blur-2xl">
        <nav className="mx-auto flex h-12 max-w-6xl items-center justify-between px-5 text-sm">
          <a href="#top" className="font-semibold tracking-tight text-[#1d1d1f]">Motion Coach AI</a>
          <div className="hidden items-center gap-7 text-xs text-[#424245] sm:flex">
            <a href="#features" className="transition hover:text-black">Features</a>
            <a href="#analyzer" className="transition hover:text-black">Analyzer</a>
            <a href="#report" className="transition hover:text-black">Report</a>
          </div>
          <button
            onClick={() => document.getElementById('analyzer')?.scrollIntoView({ behavior: 'smooth' })}
            className="rounded-full bg-[#0071e3] px-4 py-1.5 text-xs font-semibold text-white transition hover:bg-[#0077ed]"
          >
            Try now
          </button>
        </nav>
      </header>

      <main id="top">
        <section className="relative overflow-hidden bg-white">
          <div className="mx-auto grid min-h-[calc(100vh-48px)] max-w-6xl content-center gap-10 px-5 py-14 text-center">
            <motion.div
              initial={{ opacity: 0, y: 18 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6 }}
              className="mx-auto max-w-4xl"
            >
              <p className="mb-4 text-sm font-semibold text-[#6e6e73]">Motion Coach AI</p>
              <h1 className="text-5xl font-semibold tracking-tight text-[#1d1d1f] sm:text-6xl lg:text-7xl">
                Badminton form, analyzed in motion.
              </h1>
              <p className="mx-auto mt-6 max-w-2xl text-lg leading-8 text-[#6e6e73] sm:text-xl">
                Upload a clip, see your pose overlay, measure joint angles, and leave with clear coaching notes.
              </p>
              <div className="mt-8 flex flex-col items-center justify-center gap-3 sm:flex-row">
                <button
                  onClick={() => document.getElementById('analyzer')?.scrollIntoView({ behavior: 'smooth' })}
                  className="rounded-full bg-[#0071e3] px-7 py-3 text-sm font-semibold text-white transition hover:bg-[#0077ed]"
                >
                  Start analysis
                </button>
                <button
                  onClick={handleExample}
                  className="rounded-full border border-[#0071e3]/25 px-7 py-3 text-sm font-semibold text-[#0071e3] transition hover:border-[#0071e3]/50 hover:bg-[#f0f7ff]"
                >
                  View demo
                </button>
              </div>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 24, scale: 0.98 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              transition={{ duration: 0.8, delay: 0.1 }}
              className="mx-auto w-full max-w-5xl"
            >
              <img
                src={heroImage}
                alt="Motion Coach AI product preview showing badminton pose analysis on modern devices"
                className="mx-auto w-full rounded-[28px] shadow-[0_30px_90px_rgba(0,0,0,0.14)]"
              />
            </motion.div>
          </div>
        </section>

        <section id="features" className="bg-[#f5f5f7] px-5 py-16 sm:py-24">
          <div className="mx-auto max-w-6xl">
            <div className="mx-auto max-w-3xl text-center">
              <p className="text-sm font-semibold text-[#6e6e73]">Built for focused improvement</p>
              <h2 className="mt-3 text-4xl font-semibold tracking-tight sm:text-5xl">
                From raw video to useful feedback.
              </h2>
            </div>
            <div className="mt-12 grid gap-5 md:grid-cols-3">
              {[
                ['Pose overlay', 'MoveNet tracks the body frame by frame so form issues become visible.'],
                ['Live metrics', 'Knee, hip, shoulder, and elbow angles update as the clip plays.'],
                ['Coach-ready report', 'Export a simple summary that explains what to improve next.'],
              ].map(([title, text]) => (
                <div key={title} className="rounded-[28px] bg-white p-7 shadow-[0_18px_50px_rgba(0,0,0,0.05)]">
                  <h3 className="text-2xl font-semibold tracking-tight">{title}</h3>
                  <p className="mt-4 text-sm leading-7 text-[#6e6e73]">{text}</p>
                </div>
              ))}
            </div>
          </div>
        </section>

        <section id="analyzer" className="bg-[#111113] px-5 py-16 text-white sm:py-24">
          <div className="mx-auto max-w-6xl">
            <div className="mb-10 flex flex-col gap-4 md:flex-row md:items-end md:justify-between">
              <div>
                <p className="text-sm font-semibold text-[#86868b]">Analyzer</p>
                <h2 className="mt-3 text-4xl font-semibold tracking-tight sm:text-5xl">
                  Drop in a clip. Press play. Study the movement.
                </h2>
              </div>
              <div className="rounded-full bg-white/8 px-5 py-3 text-sm text-[#d2d2d7]">
                Confidence {Math.round(poseConfidence * 100)}%
              </div>
            </div>

            <div className="grid gap-6 lg:grid-cols-[0.82fr_1.18fr]">
              <div className="space-y-5">
                <div className="rounded-[28px] bg-white p-5 text-[#1d1d1f]">
                  <p className="text-xs font-semibold uppercase tracking-[0.18em] text-[#86868b]">Input</p>
                  <label className="mt-4 flex cursor-pointer items-center justify-center rounded-2xl border border-dashed border-[#d2d2d7] bg-[#f5f5f7] px-5 py-8 text-center transition hover:border-[#0071e3]/50 hover:bg-[#f0f7ff]">
                    <span>
                      <span className="block text-base font-semibold">Upload a video</span>
                      <span className="mt-2 block text-sm text-[#6e6e73]">MP4, MOV, or WebM</span>
                    </span>
                    <input className="sr-only" type="file" accept={supportedVideoTypes.join(',')} onChange={handleUpload} />
                  </label>
                  <div className="mt-4 flex gap-2">
                    <input
                      value={sourceUrl}
                      onChange={(event) => setSourceUrl(event.target.value)}
                      placeholder="Paste video URL"
                      className="min-w-0 flex-1 rounded-full border border-[#d2d2d7] bg-white px-4 py-3 text-sm outline-none transition focus:border-[#0071e3] focus:ring-4 focus:ring-[#0071e3]/10"
                    />
                    <button
                      onClick={handleLoadUrl}
                      className="rounded-full bg-[#1d1d1f] px-5 py-3 text-sm font-semibold text-white transition hover:bg-black"
                    >
                      Load
                    </button>
                  </div>
                  <button
                    onClick={handleExample}
                    className="mt-3 w-full rounded-full border border-[#d2d2d7] px-5 py-3 text-sm font-semibold text-[#0071e3] transition hover:bg-[#f5f5f7]"
                  >
                    Load demo video
                  </button>
                </div>

                <div className="rounded-[28px] bg-white/8 p-5">
                  <p className="text-xs font-semibold uppercase tracking-[0.18em] text-[#86868b]">Camera checklist</p>
                  <ul className="mt-4 space-y-3 text-sm leading-6 text-[#d2d2d7]">
                    <li>Use a side view with the full body visible.</li>
                    <li>Keep the athlete separate from the background.</li>
                    <li>Record in bright, even lighting.</li>
                    <li>Analyze one athlete at a time.</li>
                  </ul>
                </div>

                <div className="rounded-[28px] bg-white/8 p-5">
                  <p className="text-xs font-semibold uppercase tracking-[0.18em] text-[#86868b]">Status</p>
                  <p className="mt-3 text-lg font-semibold text-white">{status}</p>
                  <p className="mt-3 text-sm text-[#86868b]">Source: {getVideoSourceLabel(videoUrl)}</p>
                </div>
              </div>

              <div
                onDrop={handleDrop}
                onDragOver={(event) => {
                  event.preventDefault();
                  setDragActive(true);
                }}
                onDragLeave={() => setDragActive(false)}
                className={`overflow-hidden rounded-[32px] border ${
                  dragActive ? 'border-[#0071e3] bg-[#0071e3]/10' : 'border-white/10 bg-[#1c1c1e]'
                }`}
              >
                <div className="relative bg-black">
                  {!videoUrl && (
                    <div className="flex aspect-video flex-col items-center justify-center px-8 text-center">
                      <p className="text-sm font-semibold text-[#86868b]">Ready when your clip is.</p>
                      <h3 className="mt-3 text-3xl font-semibold tracking-tight">Drop a badminton video here.</h3>
                      <p className="mt-4 max-w-lg text-sm leading-6 text-[#86868b]">
                        The analyzer will load the pose model only after you choose a video, keeping the first page fast.
                      </p>
                    </div>
                  )}
                  {videoUrl && (
                    <>
                      {(modelLoading || !detectorRef.current) && (
                        <div className="absolute inset-0 z-20 flex items-center justify-center bg-black/55 backdrop-blur-sm">
                          <div className="rounded-full bg-white px-5 py-3 text-sm font-semibold text-[#1d1d1f]">
                            Preparing AI model
                          </div>
                        </div>
                      )}
                      <video
                        ref={videoRef}
                        src={videoUrl}
                        controls
                        muted
                        playsInline
                        crossOrigin="anonymous"
                        onTimeUpdate={handleTimeUpdate}
                        onLoadedMetadata={handleDuration}
                        className="aspect-video w-full bg-black object-contain"
                      />
                      <canvas ref={canvasRef} className="pointer-events-none absolute inset-0 h-full w-full" />
                    </>
                  )}
                </div>

                <div className="space-y-5 p-5">
                  <div className="flex items-center justify-between text-sm text-[#d2d2d7]">
                    <span>{formatTime(currentTime)} / {formatTime(duration)}</span>
                    <span>{Math.round(percent)}%</span>
                  </div>
                  <input
                    type="range"
                    min={0}
                    max={duration || 0}
                    step={0.01}
                    value={currentTime}
                    onChange={(event) => seekVideo(Number(event.target.value))}
                    className="h-1.5 w-full cursor-pointer appearance-none rounded-full bg-white/20 accent-[#0071e3]"
                  />
                  <div className="grid gap-3 sm:grid-cols-2">
                    <button
                      onClick={() => stepFrame('back')}
                      className="rounded-full bg-white/10 px-4 py-3 text-sm font-semibold text-white transition hover:bg-white/15"
                    >
                      Previous frame
                    </button>
                    <button
                      onClick={() => stepFrame('forward')}
                      className="rounded-full bg-white/10 px-4 py-3 text-sm font-semibold text-white transition hover:bg-white/15"
                    >
                      Next frame
                    </button>
                  </div>
                </div>
              </div>
            </div>

            <div className="mt-6 grid gap-4 md:grid-cols-2 xl:grid-cols-4">
              {analysisStats.map((metric) => (
                <div key={metric.key} className="rounded-[24px] bg-white p-5 text-[#1d1d1f]">
                  <div className="flex items-start justify-between gap-4">
                    <div>
                      <p className="text-xs font-semibold uppercase tracking-[0.16em] text-[#86868b]">{metric.label}</p>
                      <p className="mt-3 text-3xl font-semibold tracking-tight">{formatAngle(metric.current)}</p>
                    </div>
                    <span className="rounded-full bg-[#f5f5f7] px-3 py-1 text-xs font-semibold" style={{ color: metric.color }}>
                      {metric.current !== null ? (metric.current > 120 ? 'High' : metric.current < 70 ? 'Low' : 'Ideal') : 'Waiting'}
                    </span>
                  </div>
                  <div className="mt-4 grid grid-cols-3 gap-2 text-xs text-[#6e6e73]">
                    <span>Avg {formatAngle(metric.avg)}</span>
                    <span>Min {formatAngle(metric.min)}</span>
                    <span>Max {formatAngle(metric.max)}</span>
                  </div>
                  <div className="mt-4 h-20">
                    <ResponsiveContainer width="100%" height="100%">
                      <AreaChart data={metric.history}>
                        <XAxis dataKey="index" hide />
                        <Tooltip
                          contentStyle={{ background: '#ffffff', border: '1px solid #e8e8ed', borderRadius: 14 }}
                          formatter={(value: number) => [`${Math.round(value)} deg`, metric.label]}
                        />
                        <Area type="monotone" dataKey="value" stroke={metric.color} fill={metric.color} fillOpacity={0.16} strokeWidth={2} />
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </section>

        <section id="report" className="bg-white px-5 py-16 sm:py-24">
          <div className="mx-auto grid max-w-6xl gap-8 lg:grid-cols-[1fr_0.82fr]">
            <div>
              <p className="text-sm font-semibold text-[#6e6e73]">Coaching output</p>
              <h2 className="mt-3 text-4xl font-semibold tracking-tight sm:text-5xl">{finalReport.title}</h2>
              <p className="mt-5 max-w-2xl text-lg leading-8 text-[#6e6e73]">
                A concise summary helps athletes understand what changed, what is stable, and what to focus on next.
              </p>
              <button
                onClick={downloadReport}
                disabled={!videoUrl}
                className="mt-8 rounded-full bg-[#1d1d1f] px-7 py-3 text-sm font-semibold text-white transition hover:bg-black disabled:cursor-not-allowed disabled:bg-[#d2d2d7]"
              >
                Download report
              </button>
            </div>

            <div className="rounded-[32px] bg-[#f5f5f7] p-6">
              <p className="text-xs font-semibold uppercase tracking-[0.18em] text-[#86868b]">Live coaching</p>
              <p className="mt-4 text-xl font-semibold leading-8">{currentFeedback}</p>
              <div className="mt-6 space-y-3">
                {finalReport.bullets.map((item, index) => (
                  <div key={index} className="rounded-2xl bg-white p-4 text-sm leading-6 text-[#424245]">
                    {item}
                  </div>
                ))}
              </div>
              <div className="mt-5 grid grid-cols-2 gap-3 text-sm">
                <div className="rounded-2xl bg-white p-4">
                  <span className="block text-[#86868b]">Frames</span>
                  <strong className="mt-1 block text-xl">{metrics.knee.history.length}</strong>
                </div>
                <div className="rounded-2xl bg-white p-4">
                  <span className="block text-[#86868b]">State</span>
                  <strong className="mt-1 block text-xl">{isAnalyzing ? 'Analyzing' : 'Idle'}</strong>
                </div>
              </div>
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}
