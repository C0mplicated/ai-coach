import type * as poseDetection from '@tensorflow-models/pose-detection';

export type MetricKey = 'knee' | 'hip' | 'shoulder' | 'elbow';

export const supportedVideoTypes = ['video/mp4', 'video/mov', 'video/webm', 'video/quicktime'];
export const sampleVideoUrl = 'https://storage.googleapis.com/gtv-videos-bucket/sample/ForBiggerJoyrides.mp4';

const jointMap: Record<MetricKey, [string, string, string]> = {
  knee: ['left_hip', 'left_knee', 'left_ankle'],
  hip: ['left_shoulder', 'left_hip', 'left_knee'],
  shoulder: ['left_hip', 'left_shoulder', 'left_elbow'],
  elbow: ['left_shoulder', 'left_elbow', 'left_wrist'],
};

export const metricDefinitions: Record<MetricKey, { label: string; color: string }> = {
  knee: { label: 'Knee Angle', color: '#38bdf8' },
  hip: { label: 'Hip Angle', color: '#a78bfa' },
  shoulder: { label: 'Shoulder Angle', color: '#f472b6' },
  elbow: { label: 'Elbow Angle', color: '#34d399' },
};

export function computeAngle(a: poseDetection.Keypoint, b: poseDetection.Keypoint, c: poseDetection.Keypoint) {
  const vector = (p1: poseDetection.Keypoint, p2: poseDetection.Keypoint) => [p1.x - p2.x, p1.y - p2.y];
  const [bx, by] = vector(a, b);
  const [cx, cy] = vector(c, b);
  const dot = bx * cx + by * cy;
  const magnitude = Math.hypot(bx, by) * Math.hypot(cx, cy);
  if (!magnitude) return null;
  const radians = Math.acos(Math.max(-1, Math.min(1, dot / magnitude)));
  return (radians * 180) / Math.PI;
}

const getBestSide = (keypoints: poseDetection.Keypoint[], left: [string, string, string], right: [string, string, string]) => {
  const find = (name: string) => keypoints.find((keypoint) => keypoint.name === name);
  const resolve = (keys: [string, string, string]) => {
    const [a, b, c] = keys.map(find);
    if (!a || !b || !c || a.score == null || b.score == null || c.score == null) return null;
    return { angle: computeAngle(a, b, c), score: (a.score + b.score + c.score) / 3 };
  };

  const leftResult = resolve(left);
  const rightResult = resolve(right);
  if (!leftResult && !rightResult) return null;
  if (!leftResult) return rightResult;
  if (!rightResult) return leftResult;
  return leftResult.score >= rightResult.score ? leftResult : rightResult;
};

export function getJointAngle(metric: MetricKey, keypoints: poseDetection.Keypoint[]) {
  if (metric === 'knee') {
    const left: [string, string, string] = ['left_hip', 'left_knee', 'left_ankle'];
    const right: [string, string, string] = ['right_hip', 'right_knee', 'right_ankle'];
    return getBestSide(keypoints, left, right)?.angle ?? null;
  }
  if (metric === 'hip') {
    const left: [string, string, string] = ['left_shoulder', 'left_hip', 'left_knee'];
    const right: [string, string, string] = ['right_shoulder', 'right_hip', 'right_knee'];
    return getBestSide(keypoints, left, right)?.angle ?? null;
  }
  if (metric === 'shoulder') {
    const left: [string, string, string] = ['left_hip', 'left_shoulder', 'left_elbow'];
    const right: [string, string, string] = ['right_hip', 'right_shoulder', 'right_elbow'];
    return getBestSide(keypoints, left, right)?.angle ?? null;
  }
  if (metric === 'elbow') {
    const left: [string, string, string] = ['left_shoulder', 'left_elbow', 'left_wrist'];
    const right: [string, string, string] = ['right_shoulder', 'right_elbow', 'right_wrist'];
    return getBestSide(keypoints, left, right)?.angle ?? null;
  }
  return null;
}

export function getPoseConfidence(keypoints: poseDetection.Keypoint[]) {
  const visible = keypoints.filter((keypoint) => keypoint.score != null && keypoint.score > 0);
  if (!visible.length) return 0;
  return visible.reduce((sum, point) => sum + (point.score ?? 0), 0) / visible.length;
}

export function getCoachingFeedback({ metrics, poseConfidence }: { metrics: Record<MetricKey, { current: number | null }>; poseConfidence: number }) {
  if (poseConfidence < 0.35) {
    return 'Pose confidence is low. Try a clearer camera angle or stronger lighting.';
  }

  const currentKnee = metrics.knee.current;
  const currentHip = metrics.hip.current;

  if (currentKnee !== null && currentKnee < 70) {
    return 'Knee bend is too shallow during the lunge. Increase flexion for better power transfer.';
  }
  if (currentKnee !== null && currentKnee > 115) {
    return 'Good knee flexion, but keep your center of gravity stable as you recover.';
  }
  if (currentHip !== null && currentHip < 60) {
    return 'Try maintaining a more stable front hip position during the shot.';
  }
  return 'Joint tracking looks steady. Keep an eye on balance and posture through each frame.';
}
