"use client";

import { useState } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Loader2, X } from "lucide-react";
import { missionService } from "@/lib/api/missionService";
import { userService } from "@/lib/api/userService";
import { MISSION_TYPE_LABELS_FR } from "@/lib/utils";
import type { MissionType, User } from "@/types";

const schema = z
  .object({
    name:         z.string().min(3, "Nom trop court (3 caractères min.)"),
    company_name: z.string().min(2, "Société requise"),
    mission_type: z.enum([
      "financial_audit",
      "fraud_detection",
      "compliance_review",
      "risk_assessment",
      "internal_audit",
    ]),
    description: z.string().optional(),
    start_date:  z.string().min(1, "Date de début requise"),
    end_date:    z.string().min(1, "Date de fin requise"),
  })
  .refine((d) => d.start_date <= d.end_date, {
    message: "La date de fin doit être après le début",
    path: ["end_date"],
  });

type FormValues = z.infer<typeof schema>;

interface CreateMissionModalProps {
  open: boolean;
  onClose: () => void;
}

export function CreateMissionModal({ open, onClose }: CreateMissionModalProps) {
  const queryClient = useQueryClient();
  const [selectedAuditors, setSelectedAuditors] = useState<User[]>([]);

  const { data: auditors = [] } = useQuery({
    queryKey: ["users", "auditors"],
    queryFn:  userService.getAuditors,
    enabled:  open,
  });

  const {
    register,
    handleSubmit,
    setValue,
    reset,
    formState: { errors },
  } = useForm<FormValues>({
    resolver: zodResolver(schema),
    defaultValues: { mission_type: "financial_audit" },
  });

  const mutation = useMutation({
    mutationFn: missionService.create,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["missions"] });
      toast.success("Mission créée avec succès.");
      reset();
      setSelectedAuditors([]);
      onClose();
    },
    onError: () => {
      toast.error("Erreur lors de la création de la mission.");
    },
  });

  const onSubmit = (values: FormValues) =>
    mutation.mutate({
      ...values,
      assigned_to:       selectedAuditors[0]?.id,
      assigned_auditors: selectedAuditors.map((a) => a.id),
    });

  const handleClose = () => {
    reset();
    setSelectedAuditors([]);
    onClose();
  };

  const addAuditor = (id: string) => {
    const auditor = auditors.find((a) => a.id === id);
    if (auditor && !selectedAuditors.find((a) => a.id === id)) {
      setSelectedAuditors((prev) => [...prev, auditor]);
    }
  };

  const removeAuditor = (id: string) => {
    setSelectedAuditors((prev) => prev.filter((a) => a.id !== id));
  };

  const availableAuditors = auditors.filter(
    (a) => !selectedAuditors.find((s) => s.id === a.id)
  );

  return (
    <Dialog open={open} onOpenChange={(o) => !o && handleClose()}>
      <DialogContent className="max-w-lg">
        <DialogHeader>
          <DialogTitle>Créer une nouvelle mission</DialogTitle>
        </DialogHeader>

        <form onSubmit={handleSubmit(onSubmit)} className="space-y-4">
          {/* Mission name */}
          <div className="space-y-1.5">
            <Label htmlFor="name">Nom de la mission *</Label>
            <Input id="name" placeholder="Ex: Audit annuel 2025 — Société X" {...register("name")} />
            {errors.name && <p className="text-xs text-destructive">{errors.name.message}</p>}
          </div>

          {/* Company */}
          <div className="space-y-1.5">
            <Label htmlFor="company_name">Société *</Label>
            <Input id="company_name" placeholder="Nom de l'entité auditée" {...register("company_name")} />
            {errors.company_name && (
              <p className="text-xs text-destructive">{errors.company_name.message}</p>
            )}
          </div>

          {/* Type */}
          <div className="space-y-1.5">
            <Label>Type de mission *</Label>
            <Select
              defaultValue="financial_audit"
              onValueChange={(v) => setValue("mission_type", v as MissionType)}
            >
              <SelectTrigger>
                <SelectValue placeholder="Sélectionner un type" />
              </SelectTrigger>
              <SelectContent>
                {(Object.keys(MISSION_TYPE_LABELS_FR) as MissionType[]).map((k) => (
                  <SelectItem key={k} value={k}>
                    {MISSION_TYPE_LABELS_FR[k]}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            {errors.mission_type && (
              <p className="text-xs text-destructive">{errors.mission_type.message}</p>
            )}
          </div>

          {/* Multi-auditor assignment */}
          <div className="space-y-1.5">
            <Label>Auditeurs assignés</Label>

            {/* Selected auditor badges */}
            {selectedAuditors.length > 0 && (
              <div className="flex flex-wrap gap-1.5 rounded-md border border-border bg-muted/30 p-2">
                {selectedAuditors.map((a) => (
                  <Badge key={a.id} variant="secondary" className="gap-1 pr-1">
                    {a.name}
                    <button
                      type="button"
                      onClick={() => removeAuditor(a.id)}
                      className="ml-0.5 rounded-full hover:bg-muted-foreground/20"
                      aria-label={`Retirer ${a.name}`}
                    >
                      <X className="h-3 w-3" />
                    </button>
                  </Badge>
                ))}
              </div>
            )}

            {/* Add auditor dropdown */}
            {availableAuditors.length > 0 && (
              <Select onValueChange={addAuditor} value="">
                <SelectTrigger>
                  <SelectValue placeholder="Ajouter un auditeur…" />
                </SelectTrigger>
                <SelectContent>
                  {availableAuditors.map((a) => (
                    <SelectItem key={a.id} value={a.id}>
                      {a.name} — {a.position ?? a.email}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            )}

            {selectedAuditors.length === 0 && (
              <p className="text-xs text-muted-foreground">
                Aucun auditeur assigné. La mission sera non assignée.
              </p>
            )}
          </div>

          {/* Dates */}
          <div className="grid grid-cols-2 gap-3">
            <div className="space-y-1.5">
              <Label htmlFor="start_date">Date de début *</Label>
              <Input id="start_date" type="date" {...register("start_date")} />
              {errors.start_date && (
                <p className="text-xs text-destructive">{errors.start_date.message}</p>
              )}
            </div>
            <div className="space-y-1.5">
              <Label htmlFor="end_date">Date de fin *</Label>
              <Input id="end_date" type="date" {...register("end_date")} />
              {errors.end_date && (
                <p className="text-xs text-destructive">{errors.end_date.message}</p>
              )}
            </div>
          </div>

          {/* Description */}
          <div className="space-y-1.5">
            <Label htmlFor="description">Description</Label>
            <Textarea
              id="description"
              placeholder="Objectifs et périmètre de la mission…"
              rows={3}
              {...register("description")}
            />
          </div>

          <DialogFooter className="pt-2">
            <Button type="button" variant="outline" onClick={handleClose}>
              Annuler
            </Button>
            <Button type="submit" disabled={mutation.isPending}>
              {mutation.isPending && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
              Créer la mission
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}
