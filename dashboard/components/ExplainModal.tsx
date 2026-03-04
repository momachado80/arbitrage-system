"use client";

import { useCallback, useEffect } from "react";

interface Props {
  open: boolean;
  onClose: () => void;
  title: string;
  children: React.ReactNode;
}

export default function ExplainModal({ open, onClose, title, children }: Props) {
  const handleKey = useCallback(
    (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    },
    [onClose],
  );

  useEffect(() => {
    if (open) document.addEventListener("keydown", handleKey);
    return () => document.removeEventListener("keydown", handleKey);
  }, [open, handleKey]);

  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm"
      onClick={onClose}
    >
      <div
        className="terminal-panel max-w-lg w-full mx-4 shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-terminal-cyan text-sm font-bold tracking-wider uppercase">
            {title}
          </h3>
          <button
            onClick={onClose}
            className="text-terminal-muted hover:text-terminal-text transition-colors text-lg leading-none"
          >
            ✕
          </button>
        </div>
        <div className="text-terminal-text text-sm leading-relaxed space-y-3">
          {children}
        </div>
      </div>
    </div>
  );
}
