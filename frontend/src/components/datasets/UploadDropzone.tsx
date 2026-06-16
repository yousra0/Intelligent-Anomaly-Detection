"use client";

import { useCallback, useState } from "react";
import { cn } from "@/lib/utils";
import { UploadCloud, File, X } from "lucide-react";
import { Progress } from "@/components/ui/progress";

const ACCEPTED_TYPES = [
  "text/csv",
  "application/vnd.ms-excel",
  "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
  "text/plain",
];

const ACCEPTED_EXTENSIONS = [".csv", ".xls", ".xlsx", ".txt"];

const MAX_SIZE_MB = 500;
const MAX_SIZE_BYTES = MAX_SIZE_MB * 1024 * 1024;

interface UploadDropzoneProps {
  onFileSelect: (file: File) => void;
  uploadProgress?: number;
  uploadSpeed?: number; // bytes/s, optional
  disabled?: boolean;
  label?: string;
}

export function UploadDropzone({
  onFileSelect,
  uploadProgress,
  uploadSpeed,
  disabled = false,
  label = "Glissez votre fichier ici ou cliquez pour sélectionner",
}: UploadDropzoneProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [error, setError] = useState<string | null>(null);

  const validate = (file: File): boolean => {
    const ext = "." + file.name.split(".").pop()?.toLowerCase();
    if (!ACCEPTED_EXTENSIONS.includes(ext) && !ACCEPTED_TYPES.includes(file.type)) {
      setError(`Format non supporté. Formats acceptés: ${ACCEPTED_EXTENSIONS.join(", ")}`);
      return false;
    }
    if (file.size > MAX_SIZE_BYTES) {
      setError(`Fichier trop volumineux (max ${MAX_SIZE_MB} Mo).`);
      return false;
    }
    setError(null);
    return true;
  };

  const handleFile = useCallback(
    (file: File) => {
      if (disabled) return;
      if (!validate(file)) return;
      setSelectedFile(file);
      onFileSelect(file);
    },
    [disabled, onFileSelect]
  );

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    },
    [handleFile]
  );

  const onInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
  };

  return (
    <div className="space-y-3">
      <label
        className={cn(
          "relative flex min-h-[160px] cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed transition-colors",
          isDragging
            ? "border-pwc-orange bg-orange-50"
            : "border-border bg-muted/30 hover:border-pwc-orange hover:bg-accent/30",
          disabled && "pointer-events-none opacity-50",
          error && "border-destructive"
        )}
        onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={onDrop}
      >
        <input
          type="file"
          accept={ACCEPTED_EXTENSIONS.join(",")}
          className="sr-only"
          onChange={onInputChange}
          disabled={disabled}
        />

        {selectedFile ? (
          <div className="flex flex-col items-center gap-2 p-4 text-center">
            <File className="h-10 w-10 text-pwc-orange" />
            <span className="text-sm font-medium">{selectedFile.name}</span>
            <span className="text-xs text-muted-foreground">
              {(selectedFile.size / 1024).toFixed(0)} Ko
            </span>
            <button
              type="button"
              onClick={(e) => {
                e.preventDefault();
                setSelectedFile(null);
              }}
              className="mt-1 flex items-center gap-1 text-xs text-muted-foreground hover:text-destructive"
            >
              <X className="h-3 w-3" />
              Changer de fichier
            </button>
          </div>
        ) : (
          <div className="flex flex-col items-center gap-3 p-6 text-center">
            <UploadCloud
              className={cn("h-12 w-12", isDragging ? "text-pwc-orange" : "text-muted-foreground/60")}
            />
            <div>
              <p className="text-sm font-medium">{label}</p>
              <p className="mt-1 text-xs text-muted-foreground">
                {ACCEPTED_EXTENSIONS.join(", ")} — max {MAX_SIZE_MB} Mo
              </p>
            </div>
          </div>
        )}
      </label>

      {error && <p className="text-xs text-destructive">{error}</p>}

      {uploadProgress !== undefined && uploadProgress > 0 && (
        <div className="space-y-1">
          <div className="flex justify-between text-xs text-muted-foreground">
            <span>
              Téléversement…
              {uploadSpeed !== undefined && uploadSpeed > 0 && (
                <span className="ml-2 text-muted-foreground/70">
                  {uploadSpeed >= 1024 * 1024
                    ? `${(uploadSpeed / 1024 / 1024).toFixed(1)} Mo/s`
                    : `${(uploadSpeed / 1024).toFixed(0)} Ko/s`}
                </span>
              )}
            </span>
            <span className="font-medium">{uploadProgress}%</span>
          </div>
          <Progress value={uploadProgress} className="h-2" />
        </div>
      )}
    </div>
  );
}
