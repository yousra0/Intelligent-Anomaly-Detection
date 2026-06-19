/**
 * prisma/seed.ts
 * Seeds the database with initial demo users and sample missions.
 * Run with: npx ts-node prisma/seed.ts
 * or: npx prisma db seed
 */

import { PrismaClient } from "@prisma/client";
import * as bcrypt from "bcryptjs";

const prisma = new PrismaClient();

async function main() {
  console.log("Seeding database…");

  const password = await bcrypt.hash(process.env.DEMO_PASSWORD ?? "pwc2024", 12);

  // Create demo users
  const users = await Promise.all([
    prisma.user.upsert({
      where: { email: "auditeur@pwc.com" },
      update: {},
      create: {
        email: "auditeur@pwc.com",
        name: "Sophie Aubert",
        role: "auditor",
        phone: "+33 6 12 34 56 78",
        position: "Auditrice Senior",
        department: "Audit Financier",
        status: "active",
        password,
      },
    }),
    prisma.user.upsert({
      where: { email: "manager@pwc.com" },
      update: {},
      create: {
        email: "manager@pwc.com",
        name: "Marc Martin",
        role: "manager",
        phone: "+33 6 23 45 67 89",
        position: "Responsable Audit",
        department: "Audit Financier",
        status: "active",
        password,
      },
    }),
    prisma.user.upsert({
      where: { email: "partner@pwc.com" },
      update: {},
      create: {
        email: "partner@pwc.com",
        name: "Pierre Dupont",
        role: "partner",
        phone: "+33 6 34 56 78 90",
        position: "Associé",
        department: "Direction",
        status: "active",
        password,
      },
    }),
    prisma.user.upsert({
      where: { email: "admin@pwc.com" },
      update: {},
      create: {
        email: "admin@pwc.com",
        name: "Admin PwC",
        role: "admin",
        phone: "",
        position: "Administrateur Système",
        department: "IT",
        status: "active",
        password,
      },
    }),
  ]);

  const [auditor, manager] = users;
  console.log(`Created ${users.length} demo users`);

  // Create sample missions
  const m1 = await prisma.mission.upsert({
    where: { id: "m1-seed" },
    update: {},
    create: {
      id: "m1-seed",
      name: "Audit Annuel 2024 — Groupe Poulina",
      companyName: "Groupe Poulina",
      missionType: "financial_audit",
      description: "Audit des états financiers consolidés exercice 2024.",
      startDate: "2024-01-15",
      endDate: "2024-03-31",
      status: "completed",
      createdById: manager.id,
      assignedToId: auditor.id,
      assignments: { create: [{ userId: auditor.id }] },
    },
  });

  const m2 = await prisma.mission.upsert({
    where: { id: "m2-seed" },
    update: {},
    create: {
      id: "m2-seed",
      name: "Détection fraude transactions Q1 2025",
      companyName: "Banque de Tunisie",
      missionType: "fraud_detection",
      description: "Analyse des transactions du premier trimestre 2025.",
      startDate: "2025-04-01",
      endDate: "2025-06-30",
      status: "in_progress",
      createdById: manager.id,
      assignedToId: auditor.id,
      assignments: { create: [{ userId: auditor.id }] },
    },
  });

  console.log(`Created ${2} sample missions`);

  // Seed initial audit log entries
  await prisma.auditLog.createMany({
    skipDuplicates: true,
    data: [
      {
        action: "login",
        userId: manager.id,
        userName: manager.name,
        userRole: manager.role,
        details: "Connexion réussie",
      },
      {
        action: "mission_create",
        userId: manager.id,
        userName: manager.name,
        userRole: manager.role,
        missionId: m1.id,
        missionName: m1.name,
        details: `Mission créée : "${m1.name}"`,
      },
      {
        action: "mission_create",
        userId: manager.id,
        userName: manager.name,
        userRole: manager.role,
        missionId: m2.id,
        missionName: m2.name,
        details: `Mission créée : "${m2.name}"`,
      },
    ],
  });

  console.log("Seed completed successfully.");
}

main()
  .catch((e) => { console.error(e); process.exit(1); })
  .finally(async () => { await prisma.$disconnect(); });
