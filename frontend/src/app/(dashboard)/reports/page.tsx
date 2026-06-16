"use client";

import { FileText } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";

export default function ReportsPage() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Rapports</h1>
        <p className="mt-0.5 text-sm text-muted-foreground">
          Accédez aux rapports depuis la page d'analyse d'une mission.
        </p>
      </div>
      <Card>
        <CardContent className="flex flex-col items-center justify-center py-16 text-center">
          <FileText className="h-12 w-12 text-muted-foreground/30" />
          <p className="mt-4 font-medium text-muted-foreground">
            Les rapports sont générés depuis l'analyse
          </p>
          <p className="mt-1 text-sm text-muted-foreground">
            Ouvrez une mission, lancez une analyse, puis utilisez l'onglet Rapport
            pour générer les fichiers PDF et DOCX via le backend FastAPI.
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
