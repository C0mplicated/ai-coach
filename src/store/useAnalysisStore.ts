import { create } from 'zustand';
import type { MetricKey } from '../lib/poseUtils';

export interface JointStats {
  history: number[];
  current: number | null;
  avg: number;
  min: number;
  max: number;
}

export interface AnalysisState {
  videoUrl: string;
  feedback: string;
  status: string;
  poseConfidence: number;
  isAnalyzing: boolean;
  metrics: Record<MetricKey, JointStats>;
  setVideoUrl: (url: string) => void;
  pushAngle: (key: MetricKey, value: number) => void;
  resetAnalysis: () => void;
  setPoseConfidence: (value: number) => void;
  setFeedback: (value: string) => void;
  setStatus: (value: string) => void;
  setAnalyzing: (value: boolean) => void;
}

const initialStats = (): JointStats => ({
  history: [],
  current: null,
  avg: 0,
  min: 0,
  max: 0,
});

const initialMetrics: Record<MetricKey, JointStats> = {
  knee: initialStats(),
  hip: initialStats(),
  shoulder: initialStats(),
  elbow: initialStats(),
};

const updateStats = (history: number[]): JointStats => {
  const current = history[history.length - 1] ?? null;
  const avg = history.length ? history.reduce((acc, value) => acc + value, 0) / history.length : 0;
  const min = history.length ? Math.min(...history) : 0;
  const max = history.length ? Math.max(...history) : 0;
  return { history, current, avg, min, max };
};

export const useAnalysisStore = create<AnalysisState>((set) => ({
  videoUrl: '',
  feedback: 'Upload a video or paste a shared link.',
  status: 'Waiting for a video input.',
  poseConfidence: 0,
  isAnalyzing: false,
  metrics: initialMetrics,
  setVideoUrl: (url) => set(() => ({ videoUrl: url })),
  pushAngle: (key, value) =>
    set((state) => {
      const history = [...state.metrics[key].history, value].slice(-64);
      return {
        metrics: {
          ...state.metrics,
          [key]: updateStats(history),
        },
      };
    }),
  resetAnalysis: () =>
    set(() => ({
      feedback: 'Upload a video or paste a shared link.',
      status: 'Waiting for a video input.',
      poseConfidence: 0,
      isAnalyzing: false,
      metrics: initialMetrics,
    })),
  setPoseConfidence: (value) => set(() => ({ poseConfidence: value })),
  setFeedback: (value) => set(() => ({ feedback: value })),
  setStatus: (value) => set(() => ({ status: value })),
  setAnalyzing: (value) => set(() => ({ isAnalyzing: value })),
}));
