/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState, useEffect, useRef } from 'react';
import { GoogleGenAI, Modality } from "@google/genai";
import { Mic, MicOff, Phone, PhoneOff, Loader2 } from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import { float32ToInt16, int16ToFloat32, arrayBufferToBase64, base64ToArrayBuffer } from './lib/audio-utils';

export default function App() {
  const [status, setStatus] = useState<'idle' | 'connecting' | 'active' | 'error'>('idle');
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  const audioContextRef = useRef<AudioContext | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const sessionRef = useRef<any>(null);

  // URL parameters for personalization
  const params = new URLSearchParams(window.location.search);
  const name = params.get('name') || 'Candidate';
  const companyName = params.get('companyName') || params.get('Company') || 'our recruitment firm';
  const industriesServed = params.get('industriesServed') || 'various sectors';
  const jobTitle = params.get('jobTitle') || 'this';

  const audioQueueRef = useRef<Float32Array[]>([]);
  const isPlayingRef = useRef(false);
  const nextStartTimeRef = useRef<number>(0);

  const startConversation = async () => {
    try {
      setStatus('connecting');
      setError(null);

      // 1. Initialize Audio
      // Using 24000Hz as it's the standard output rate for Gemini audio models, 
      // which prevents the "slow motion" effect.
      audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 24000 });
      streamRef.current = await navigator.mediaDevices.getUserMedia({ audio: true });
      
      const source = audioContextRef.current.createMediaStreamSource(streamRef.current);
      const processor = audioContextRef.current.createScriptProcessor(4096, 1, 1);
      
      source.connect(processor);
      processor.connect(audioContextRef.current.destination);

      // 2. Initialize Gemini Live API
      const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
      
      const sessionPromise = ai.live.connect({
        model: "gemini-3.1-flash-live-preview",
        callbacks: {
          onopen: () => {
            setStatus('active');
            console.log('Live API connection opened');
          },
          onmessage: async (message) => {
            // Handle audio output
            const base64Audio = message.serverContent?.modelTurn?.parts?.[0]?.inlineData?.data;
            if (base64Audio) {
              const arrayBuffer = base64ToArrayBuffer(base64Audio);
              const int16Data = new Int16Array(arrayBuffer);
              const float32Data = int16ToFloat32(int16Data);
              audioQueueRef.current.push(float32Data);
              processAudioQueue();
            }

            // Handle interruption
            if (message.serverContent?.interrupted) {
              audioQueueRef.current = [];
              nextStartTimeRef.current = 0;
              setIsSpeaking(false);
            }

            // Check if model is speaking
            if (message.serverContent?.modelTurn) {
              setIsSpeaking(true);
            }
          },
          onclose: () => {
            setStatus('idle');
            stopConversation();
          },
          onerror: (err) => {
            console.error('Live API error:', err);
            setError('Connection error. Please try again.');
            setStatus('error');
            stopConversation();
          }
        },
        config: {
          responseModalities: [Modality.AUDIO],
          speechConfig: {
            voiceConfig: { prebuiltVoiceConfig: { voiceName: "Puck" } },
          },
          systemInstruction: `1. Agent Identity & Tone
              You are a professional recruiter conducting a prescreen call representing ${companyName}.
              You are:
              Conversational
              Confident
              Curious
              Not robotic
              Goal: Understand the candidate deeply, decide fit, and position next steps.
              2. Opening (Intro + Context)
              Clearly introduce yourself and the company.
              Then give light context (NOT a hard job pitch): "We're currently working on roles in ${industriesServed}..."
              Avoid: Over-explaining or selling too early.
              3. Call Control Rule
              Always: Ask questions FIRST.
              Do NOT let the candidate derail into "tell me about the job".
              If they ask early: Politely say you'll cover that after understanding them better.
              4. Core Discovery (Main Section)
              Background Exploration: "Walk me through your experience." Go beyond the CV: explore roles, responsibilities, and achievements.
              Motivation & Intent: Why they applied / are exploring, and what they're looking for next.
              Preferences & Fit Signals: Type of role, preferred work environment, and career direction.
              Optional Skill Probing (Adaptive): Only if relevant, ask deeper about key skills for a ${jobTitle} position. This should feel natural, not checklist-based.
              5. Conversation Style Rules
              Ask open-ended questions.
              You must drive the conversation forward. When the candidate finishes answering, it is your responsibility to immediately acknowledge it and ask the next question.
              Do NOT rapid-fire questions. Ask only one simple question at a time.
              Make it feel like a natural conversation, not an interview.
              6. Rapport Layer (Subtle but critical)
              Be warm.
              Show interest.
              Acknowledge answers briefly before your next question.
              This runs in the background of all steps.
              7. Fit Positioning (Soft "Sell")
              After understanding the candidate:
              Briefly say: "Based on what you said, you'd likely be a strong fit for ${jobTitle} roles..."
              Avoid a full job pitch
              8. Close (Light Conversion)
              Ask if they're open to moving forward or being represented. Keep it simple and natural.
              9. Ending
              Set expectations: Next steps, follow-up, and end warmly.
              10. Voice API Technical Constraints (CRITICAL)
              Always Respond: Never generate an empty or silent turn. You must always reply vocally when the user finishes speaking.
              Pace & Conciseness: Keep your responses very concise and snappy. Avoid long-winded explanations so the audio streams quickly.
              Turn-Taking: Respond immediately to the user. Do not wait or pause artificially. Keep the energy up and maintain a fast-paced conversational tempo.
                  `,
        },
      });

      sessionRef.current = await sessionPromise;

      // 3. Handle Microphone Input
      processor.onaudioprocess = (e) => {
        if (status === 'active' || sessionRef.current) {
          const inputData = e.inputBuffer.getChannelData(0);
          const int16Data = float32ToInt16(inputData);
          const base64Data = arrayBufferToBase64(int16Data.buffer);
          
          sessionRef.current.sendRealtimeInput({
            audio: { data: base64Data, mimeType: 'audio/pcm;rate=24000' }
          });
        }
      };

    } catch (err: any) {
      console.error('Failed to start conversation:', err);
      setError(err.message || 'Failed to access microphone or connect to AI.');
      setStatus('error');
    }
  };
  const processAudioQueue = () => {
    if (!audioContextRef.current || audioQueueRef.current.length === 0) return;

    const currentTime = audioContextRef.current.currentTime;
    
    // If we're behind or just starting, reset nextStartTime to slightly into the future
    if (nextStartTimeRef.current < currentTime) {
      nextStartTimeRef.current = currentTime + 0.1; // 100ms lookahead for stability
    }

    // Schedule up to 1 second into the future
    while (audioQueueRef.current.length > 0 && nextStartTimeRef.current < currentTime + 1.0) {
      const audioData = audioQueueRef.current.shift()!;
      const buffer = audioContextRef.current.createBuffer(1, audioData.length, 24000);
      buffer.getChannelData(0).set(audioData);
      
      const source = audioContextRef.current.createBufferSource();
      source.buffer = buffer;
      source.connect(audioContextRef.current.destination);
      
      source.start(nextStartTimeRef.current);
      nextStartTimeRef.current += buffer.duration;
      
      setIsSpeaking(true);
      isPlayingRef.current = true;

      source.onended = () => {
        if (audioQueueRef.current.length === 0 && nextStartTimeRef.current <= audioContextRef.current!.currentTime) {
          setIsSpeaking(false);
          isPlayingRef.current = false;
        }
      };
    }
  };

  const stopConversation = () => {
    if (sessionRef.current) {
      sessionRef.current.close();
      sessionRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }
    setStatus('idle');
    setIsSpeaking(false);
    audioQueueRef.current = [];
  };

  return (
    <div className="min-h-screen bg-bg-dark flex flex-col items-center justify-center p-6 relative overflow-hidden">
      {/* Mesh Gradient Background */}
      <div className="background-blobs" />

      <div className="glass-container z-10 w-full max-w-[640px] px-10 py-[60px] text-center">
        {/* Logo Mark */}
        <motion.div 
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-[12px] uppercase tracking-[0.3em] text-accent-blue font-bold mb-8"
        >
          Voice Intelligence System
        </motion.div>

        {/* Header Section */}
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="space-y-6"
        >
          <h1 className="text-[48px] font-light leading-[1.1] tracking-tight text-white">
            <span className="font-semibold">{name}</span>, Ready For The <br />
            Pre-Screen Demo?
          </h1>
          <p className="text-[16px] text-text-secondary max-w-[400px] mx-auto leading-relaxed">
            You are about to enter a live screening session powered by high-fidelity voice AI on behalf of{" "}
            <span className="text-white font-medium">{companyName}</span>.
          </p>
        </motion.div>

        {/* Main Interaction Area */}
        <div className="relative flex flex-col items-center justify-center py-12">
          {/* Pulse Animation for Speaking */}
          <AnimatePresence>
            {(status === 'active' || status === 'connecting') && (
              <motion.div
                initial={{ scale: 0.8, opacity: 0 }}
                animate={{ 
                  scale: isSpeaking ? [1, 1.2, 1] : 1,
                  opacity: 1 
                }}
                exit={{ scale: 0.8, opacity: 0 }}
                transition={{ 
                  scale: { repeat: Infinity, duration: 1.5, ease: "easeInOut" },
                  opacity: { duration: 0.3 }
                }}
                className={`absolute w-48 h-48 rounded-full blur-3xl ${isSpeaking ? 'bg-blue-500/20' : 'bg-white/5'}`}
              />
            )}
          </AnimatePresence>

          {/* Action Button */}
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={status === 'idle' || status === 'error' ? startConversation : stopConversation}
            className={`relative z-10 w-[100px] h-[100px] rounded-full flex items-center justify-center transition-all duration-300 ${
              status === 'active' 
                ? 'bg-red-500 shadow-[0_0_0_10px_rgba(239,68,68,0.1)]' 
                : status === 'connecting'
                ? 'bg-blue-400 shadow-[0_0_0_10px_rgba(96,165,250,0.1)]'
                : 'bg-accent-blue shadow-[0_0_0_10px_rgba(59,130,246,0.1)]'
            } text-white`}
            disabled={status === 'connecting'}
          >
            {status === 'connecting' ? (
              <Loader2 className="w-8 h-8 animate-spin" />
            ) : status === 'active' ? (
              <PhoneOff className="w-8 h-8" />
            ) : (
              <Mic className="w-8 h-8" />
            )}
          </motion.button>

          {/* Status Text */}
          <motion.div 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="mt-5 h-6"
          >
            {status === 'connecting' && <p className="text-accent-blue text-[14px] font-medium animate-pulse">Connecting...</p>}
            {status === 'active' && <p className="text-green-400 text-[14px] font-medium">Live Session</p>}
            {status === 'error' && <p className="text-red-400 text-[14px] font-medium">{error}</p>}
            {status === 'idle' && <p className="text-white text-[14px] font-medium">Click to start</p>}
          </motion.div>
        </div>
      </div>

      {/* Metadata Section */}
      <motion.div 
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.8 }}
        className="absolute bottom-10 flex gap-6 text-[11px] text-text-secondary uppercase tracking-[0.1em]"
      >
        <div className="flex items-center gap-2">
          <span className={`w-1.5 h-1.5 rounded-full ${status === 'active' ? 'bg-green-500 shadow-[0_0_8px_rgba(34,197,94,0.5)]' : 'bg-slate-600'}`} />
          Agent {status === 'active' ? 'Active' : 'Ready'}
        </div>
        <div className="hidden md:block">Entity: {companyName}</div>
        <div className="hidden md:block">Subject: {name}</div>
      </motion.div>
    </div>
  );
}
