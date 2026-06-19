"use client";

import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useRouter } from "next/navigation";
import { useEffect } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import { toast } from "sonner";
import { userService } from "@/lib/api/userService";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { usePermissions } from "@/lib/hooks/usePermissions";
import { ROLE_LABELS_FR } from "@/lib/utils";
import { Loader2, Plus, MoreHorizontal, Search } from "lucide-react";
import type { User, UserRole, CreateUserPayload, UpdateUserPayload } from "@/types";

const ROLE_BADGE: Record<UserRole, string> = {
  admin:   "bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400",
  manager: "bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400",
  partner: "bg-purple-100 text-purple-700 dark:bg-purple-900/30 dark:text-purple-400",
  auditor: "bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400",
};

const createSchema = z.object({
  first_name: z.string().min(2, "Prénom requis"),
  last_name:  z.string().min(2, "Nom requis"),
  email:      z.string().email("Email invalide").endsWith("@pwc.com", "Email @pwc.com requis"),
  phone:      z.string().optional(),
  position:   z.string().optional(),
  department: z.string().optional(),
  password:   z.string().min(6, "6 caractères minimum"),
  role:       z.enum(["auditor", "manager", "partner", "admin"]),
});

const editSchema = z.object({
  first_name: z.string().min(2, "Prénom requis"),
  last_name:  z.string().min(2, "Nom requis"),
  email:      z.string().email("Email invalide"),
  phone:      z.string().optional(),
  position:   z.string().optional(),
  department: z.string().optional(),
  role:       z.enum(["auditor", "manager", "partner", "admin"]),
});

const resetPassSchema = z.object({
  password: z.string().min(6, "6 caractères minimum"),
  confirm:  z.string(),
}).refine((d) => d.password === d.confirm, {
  message: "Les mots de passe ne correspondent pas",
  path: ["confirm"],
});

type CreateForm = z.infer<typeof createSchema>;
type EditForm   = z.infer<typeof editSchema>;
type ResetForm  = z.infer<typeof resetPassSchema>;

export default function AdminUsersPage() {
  const { can } = usePermissions();
  const router  = useRouter();
  const qc      = useQueryClient();

  const [search, setSearch]         = useState("");
  const [createOpen, setCreateOpen] = useState(false);
  const [editUser, setEditUser]     = useState<User | null>(null);
  const [resetUser, setResetUser]   = useState<User | null>(null);

  useEffect(() => {
    if (!can("admin.users")) router.replace("/missions");
  }, [can, router]);

  const { data: users = [], isLoading } = useQuery({
    queryKey: ["users", "all"],
    queryFn:  userService.getAll,
    enabled:  can("admin.users"),
  });

  const createMut = useMutation({
    mutationFn: (payload: CreateUserPayload) => userService.create(payload),
    onSuccess:  () => { qc.invalidateQueries({ queryKey: ["users"] }); toast.success("Utilisateur créé."); setCreateOpen(false); },
    onError:    (e: { response?: { data?: { error?: string } } }) =>
                  toast.error(e.response?.data?.error ?? "Erreur lors de la création."),
  });

  const editMut = useMutation({
    mutationFn: ({ id, payload }: { id: string; payload: UpdateUserPayload }) =>
                  userService.update(id, payload),
    onSuccess:  () => { qc.invalidateQueries({ queryKey: ["users"] }); toast.success("Utilisateur mis à jour."); setEditUser(null); },
    onError:    () => toast.error("Erreur lors de la mise à jour."),
  });

  const statusMut = useMutation({
    mutationFn: ({ id, status }: { id: string; status: "active" | "inactive" }) =>
                  userService.setStatus(id, status),
    onSuccess:  () => { qc.invalidateQueries({ queryKey: ["users"] }); toast.success("Statut mis à jour."); },
    onError:    () => toast.error("Erreur lors de la mise à jour du statut."),
  });

  const resetMut = useMutation({
    mutationFn: ({ id, password }: { id: string; password: string }) =>
                  userService.resetPassword(id, password),
    onSuccess:  () => { toast.success("Mot de passe réinitialisé."); setResetUser(null); },
    onError:    () => toast.error("Erreur lors de la réinitialisation."),
  });

  if (!can("admin.users")) return null;

  const filtered = users.filter((u) =>
    !search ||
    u.name.toLowerCase().includes(search.toLowerCase()) ||
    u.email.toLowerCase().includes(search.toLowerCase()) ||
    (u.department ?? "").toLowerCase().includes(search.toLowerCase())
  );

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Header */}
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="text-2xl font-bold text-foreground">Gestion des utilisateurs</h1>
          <p className="mt-0.5 text-sm text-muted-foreground">
            {users.length} utilisateur{users.length !== 1 ? "s" : ""}
          </p>
        </div>
        {can("user.create") && (
          <Button className="gap-2 self-start" onClick={() => setCreateOpen(true)}>
            <Plus className="h-4 w-4" />
            Créer un auditeur
          </Button>
        )}
      </div>

      {/* Role distribution */}
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {(["auditor", "manager", "partner", "admin"] as UserRole[]).map((r) => (
          <Card key={r}>
            <CardContent className="pt-4">
              <p className="text-xs text-muted-foreground">{ROLE_LABELS_FR[r]}</p>
              <p className="mt-1 text-2xl font-bold">
                {isLoading ? "—" : users.filter((u) => u.role === r).length}
              </p>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Search + table */}
      <Card>
        <CardHeader className="pb-3">
          <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
            <CardTitle className="text-base">Liste des utilisateurs</CardTitle>
            <div className="relative sm:w-64">
              <Search className="absolute left-3 top-2.5 h-4 w-4 text-muted-foreground" />
              <Input
                placeholder="Rechercher…"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                className="pl-9"
              />
            </div>
          </div>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <div className="space-y-2">
              {Array.from({ length: 4 }).map((_, i) => (
                <Skeleton key={i} className="h-14 rounded-lg" />
              ))}
            </div>
          ) : (
            <div className="table-responsive">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-border bg-muted/30">
                    {["Utilisateur", "Email", "Poste", "Département", "Rôle", "Statut", ""].map((h) => (
                      <th key={h} className="px-4 py-2.5 text-left text-xs font-semibold text-muted-foreground whitespace-nowrap">
                        {h}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-border">
                  {filtered.map((u) => (
                    <tr key={u.id} className="hover:bg-muted/20">
                      <td className="px-4 py-3 whitespace-nowrap">
                        <div className="flex items-center gap-2">
                          <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-pwc-orange text-xs font-bold text-white">
                            {u.name.split(" ").map((n) => n[0]).join("").slice(0, 2).toUpperCase()}
                          </div>
                          <span className="font-medium">{u.name}</span>
                        </div>
                      </td>
                      <td className="px-4 py-3 text-muted-foreground">{u.email}</td>
                      <td className="px-4 py-3 text-muted-foreground">{u.position ?? "—"}</td>
                      <td className="px-4 py-3 text-muted-foreground">{u.department ?? "—"}</td>
                      <td className="px-4 py-3">
                        <span className={`rounded-full px-2.5 py-1 text-xs font-medium ${ROLE_BADGE[u.role as UserRole] ?? ""}`}>
                          {ROLE_LABELS_FR[u.role as UserRole]}
                        </span>
                      </td>
                      <td className="px-4 py-3">
                        {u.status === "inactive" ? (
                          <span className="flex items-center gap-1.5 text-xs text-muted-foreground">
                            <span className="h-1.5 w-1.5 rounded-full bg-gray-400" />
                            Inactif
                          </span>
                        ) : (
                          <span className="flex items-center gap-1.5 text-xs text-green-700 dark:text-green-400">
                            <span className="h-1.5 w-1.5 rounded-full bg-green-500" />
                            Actif
                          </span>
                        )}
                      </td>
                      <td className="px-4 py-3">
                        <DropdownMenu>
                          <DropdownMenuTrigger asChild>
                            <Button variant="ghost" size="icon" className="h-7 w-7">
                              <MoreHorizontal className="h-4 w-4" />
                            </Button>
                          </DropdownMenuTrigger>
                          <DropdownMenuContent align="end">
                            {can("user.edit") && (
                              <DropdownMenuItem onClick={() => setEditUser(u)}>
                                Modifier
                              </DropdownMenuItem>
                            )}
                            <DropdownMenuItem onClick={() => setResetUser(u)}>
                              Réinitialiser le mot de passe
                            </DropdownMenuItem>
                            {can("user.disable") && (
                              <>
                                <DropdownMenuSeparator />
                                {u.status === "inactive" ? (
                                  <DropdownMenuItem
                                    onClick={() => statusMut.mutate({ id: u.id, status: "active" })}
                                  >
                                    Activer
                                  </DropdownMenuItem>
                                ) : (
                                  <DropdownMenuItem
                                    className="text-red-600 focus:text-red-600"
                                    onClick={() => statusMut.mutate({ id: u.id, status: "inactive" })}
                                  >
                                    Désactiver
                                  </DropdownMenuItem>
                                )}
                              </>
                            )}
                          </DropdownMenuContent>
                        </DropdownMenu>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {filtered.length === 0 && (
                <p className="py-8 text-center text-sm text-muted-foreground">
                  Aucun utilisateur ne correspond à la recherche.
                </p>
              )}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Create modal */}
      <CreateUserModal
        open={createOpen}
        onClose={() => setCreateOpen(false)}
        onSubmit={(v) => createMut.mutate(v)}
        isPending={createMut.isPending}
        canManageRoles={can("user.manage_roles")}
      />

      {/* Edit modal */}
      {editUser && (
        <EditUserModal
          user={editUser}
          onClose={() => setEditUser(null)}
          onSubmit={(v) => editMut.mutate({ id: editUser.id, payload: v })}
          isPending={editMut.isPending}
          canManageRoles={can("user.manage_roles")}
        />
      )}

      {/* Reset password modal */}
      {resetUser && (
        <ResetPasswordModal
          user={resetUser}
          onClose={() => setResetUser(null)}
          onSubmit={(p) => resetMut.mutate({ id: resetUser.id, password: p })}
          isPending={resetMut.isPending}
        />
      )}
    </div>
  );
}

// ─── Create User Modal ───────────────────────────────────────────────────────

function CreateUserModal({
  open, onClose, onSubmit, isPending, canManageRoles,
}: {
  open: boolean;
  onClose: () => void;
  onSubmit: (v: CreateUserPayload) => void;
  isPending: boolean;
  canManageRoles: boolean;
}) {
  const { register, handleSubmit, setValue, reset, formState: { errors } } = useForm<CreateForm>({
    resolver: zodResolver(createSchema),
    defaultValues: { role: "auditor" },
  });

  const handleClose = () => { reset(); onClose(); };

  return (
    <Dialog open={open} onOpenChange={(o) => !o && handleClose()}>
      <DialogContent className="max-w-lg">
        <DialogHeader>
          <DialogTitle>Créer un utilisateur</DialogTitle>
        </DialogHeader>
        <form onSubmit={handleSubmit(onSubmit)} className="space-y-4">
          <div className="grid grid-cols-2 gap-3">
            <FormField label="Prénom *" error={errors.first_name?.message}>
              <Input placeholder="Sophie" {...register("first_name")} />
            </FormField>
            <FormField label="Nom *" error={errors.last_name?.message}>
              <Input placeholder="Aubert" {...register("last_name")} />
            </FormField>
          </div>
          <FormField label="Email *" error={errors.email?.message}>
            <Input type="email" placeholder="s.aubert@pwc.com" {...register("email")} />
          </FormField>
          <div className="grid grid-cols-2 gap-3">
            <FormField label="Téléphone" error={errors.phone?.message}>
              <Input placeholder="+33 6 00 00 00 00" {...register("phone")} />
            </FormField>
            <FormField label="Rôle *" error={errors.role?.message}>
              <Select
                defaultValue="auditor"
                onValueChange={(v) => setValue("role", v as UserRole)}
                disabled={!canManageRoles}
              >
                <SelectTrigger><SelectValue /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="auditor">{ROLE_LABELS_FR.auditor}</SelectItem>
                  <SelectItem value="manager">{ROLE_LABELS_FR.manager}</SelectItem>
                  <SelectItem value="partner">{ROLE_LABELS_FR.partner}</SelectItem>
                  {canManageRoles && <SelectItem value="admin">{ROLE_LABELS_FR.admin}</SelectItem>}
                </SelectContent>
              </Select>
            </FormField>
          </div>
          <div className="grid grid-cols-2 gap-3">
            <FormField label="Poste">
              <Input placeholder="Auditeur Senior" {...register("position")} />
            </FormField>
            <FormField label="Département">
              <Input placeholder="Audit Financier" {...register("department")} />
            </FormField>
          </div>
          <FormField label="Mot de passe *" error={errors.password?.message}>
            <Input type="password" placeholder="••••••••" {...register("password")} />
          </FormField>
          <DialogFooter className="pt-2">
            <Button type="button" variant="outline" onClick={handleClose}>Annuler</Button>
            <Button type="submit" disabled={isPending}>
              {isPending && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
              Créer
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}

// ─── Edit User Modal ─────────────────────────────────────────────────────────

function EditUserModal({
  user, onClose, onSubmit, isPending, canManageRoles,
}: {
  user: User;
  onClose: () => void;
  onSubmit: (v: UpdateUserPayload) => void;
  isPending: boolean;
  canManageRoles: boolean;
}) {
  const nameParts = user.name.split(" ");
  const { register, handleSubmit, setValue, formState: { errors } } = useForm<EditForm>({
    resolver: zodResolver(editSchema),
    defaultValues: {
      first_name: nameParts[0] ?? "",
      last_name:  nameParts.slice(1).join(" ") ?? "",
      email:      user.email,
      phone:      user.phone ?? "",
      position:   user.position ?? "",
      department: user.department ?? "",
      role:       user.role,
    },
  });

  return (
    <Dialog open onOpenChange={(o) => !o && onClose()}>
      <DialogContent className="max-w-lg">
        <DialogHeader>
          <DialogTitle>Modifier — {user.name}</DialogTitle>
        </DialogHeader>
        <form onSubmit={handleSubmit(onSubmit)} className="space-y-4">
          <div className="grid grid-cols-2 gap-3">
            <FormField label="Prénom *" error={errors.first_name?.message}>
              <Input {...register("first_name")} />
            </FormField>
            <FormField label="Nom *" error={errors.last_name?.message}>
              <Input {...register("last_name")} />
            </FormField>
          </div>
          <FormField label="Email *" error={errors.email?.message}>
            <Input type="email" {...register("email")} />
          </FormField>
          <div className="grid grid-cols-2 gap-3">
            <FormField label="Téléphone">
              <Input {...register("phone")} />
            </FormField>
            <FormField label="Rôle *" error={errors.role?.message}>
              <Select
                defaultValue={user.role}
                onValueChange={(v) => setValue("role", v as UserRole)}
                disabled={!canManageRoles}
              >
                <SelectTrigger><SelectValue /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="auditor">{ROLE_LABELS_FR.auditor}</SelectItem>
                  <SelectItem value="manager">{ROLE_LABELS_FR.manager}</SelectItem>
                  <SelectItem value="partner">{ROLE_LABELS_FR.partner}</SelectItem>
                  {canManageRoles && <SelectItem value="admin">{ROLE_LABELS_FR.admin}</SelectItem>}
                </SelectContent>
              </Select>
            </FormField>
          </div>
          <div className="grid grid-cols-2 gap-3">
            <FormField label="Poste">
              <Input {...register("position")} />
            </FormField>
            <FormField label="Département">
              <Input {...register("department")} />
            </FormField>
          </div>
          <DialogFooter className="pt-2">
            <Button type="button" variant="outline" onClick={onClose}>Annuler</Button>
            <Button type="submit" disabled={isPending}>
              {isPending && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
              Enregistrer
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}

// ─── Reset Password Modal ─────────────────────────────────────────────────────

function ResetPasswordModal({
  user, onClose, onSubmit, isPending,
}: {
  user: User;
  onClose: () => void;
  onSubmit: (password: string) => void;
  isPending: boolean;
}) {
  const { register, handleSubmit, formState: { errors } } = useForm<ResetForm>({
    resolver: zodResolver(resetPassSchema),
  });

  return (
    <Dialog open onOpenChange={(o) => !o && onClose()}>
      <DialogContent className="max-w-sm">
        <DialogHeader>
          <DialogTitle>Réinitialiser le mot de passe</DialogTitle>
        </DialogHeader>
        <p className="text-sm text-muted-foreground">Utilisateur : <strong>{user.name}</strong></p>
        <form onSubmit={handleSubmit((v) => onSubmit(v.password))} className="space-y-4">
          <FormField label="Nouveau mot de passe *" error={errors.password?.message}>
            <Input type="password" placeholder="••••••••" {...register("password")} />
          </FormField>
          <FormField label="Confirmer *" error={errors.confirm?.message}>
            <Input type="password" placeholder="••••••••" {...register("confirm")} />
          </FormField>
          <DialogFooter className="pt-2">
            <Button type="button" variant="outline" onClick={onClose}>Annuler</Button>
            <Button type="submit" disabled={isPending}>
              {isPending && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
              Réinitialiser
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}

// ─── Shared FormField helper ──────────────────────────────────────────────────

function FormField({
  label, error, children,
}: {
  label: string;
  error?: string;
  children: React.ReactNode;
}) {
  return (
    <div className="space-y-1.5">
      <Label>{label}</Label>
      {children}
      {error && <p className="text-xs text-destructive">{error}</p>}
    </div>
  );
}
